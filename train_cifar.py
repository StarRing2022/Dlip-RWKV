import torch
import torch.nn.functional as F
import numpy as np
import os
from omegaconf import OmegaConf

from ringrwkv.configuration_rwkv_world import RwkvConfig
from ringrwkv.rwkv_tokenizer import TRIE_TOKENIZER
from ringrwkv.modehf_world import RwkvForCausalLM

from model.model import DLIP
#from utils.simple_tokenizer import SimpleTokenizer #bpe
from utils.custom_schedulers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from utils.util import set_seed, mkdir, load_config_file
from utils.logger import setup_logger

from torch.optim import Adam, AdamW # both are same but AdamW has a default weight decay

import argparse
from dataset import CIFAR100 , CIFAR10
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

TRAINER_CONFIG_PATH = 'config/train_config.yaml'
MODEL_CONFIG_PATH = 'config/model_config.yaml'

train_config = load_config_file(TRAINER_CONFIG_PATH)
model_config = load_config_file(MODEL_CONFIG_PATH)

config = OmegaConf.merge(train_config, model_config)
logger = setup_logger("DLIP_CIFAR_TRAIN", config.logs, 0, filename = "training_logs.txt")

def train(config, train_dataset, model):
    '''
    Trains the model.
    '''
    
    config.train_batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)    
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=config.train_batch_size,shuffle=True,num_workers=0,drop_last=True)
    #print(len(train_dataloader))

    # total training iterations
    t_total = len(train_dataloader) // config.gradient_accumulation_steps \
                * config.num_train_epochs
    
    optimizer = AdamW(model.parameters(), lr=config.optimizer.params.lr, eps=config.optimizer.params.eps, weight_decay=config.optimizer.params.weight_decay)

    # Warmup iterations = 20% of total iterations
    num_warmup_steps = int(0.20 * t_total)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps= num_warmup_steps, num_training_steps= t_total)

    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    model = model.to(torch.device(config.device))
    model.train()

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Number of GPUs = %d", config.n_gpu)

    logger.info("  Batch size per GPU = %d", config.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   config.train_batch_size * config.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    if scheduler:
        logger.info("  warmup steps = %d", num_warmup_steps)


    global_step, global_loss, global_acc =1,  0.0, 0.0
    model.zero_grad()

    for epoch in range(int(config.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            
            input_images, input_texts, input_caption = batch

            input_images = input_images.to(torch.device(config.device))
            input_texts = input_texts.to(torch.device(config.device))

            #print(input_texts)
            
            image_features, text_features = model.forward(input_images, input_texts)

            #print(image_features.shape,text_features.shape) #[batchsize,embed_dim=1024]

            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            if config.n_gpu == 1:
                logit_scale = model.logit_scale.exp()
            elif config.n_gpu > 1:
                logit_scale = model.module.logit_scale.exp()

            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()

            labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

            image_loss = F.cross_entropy(logits_per_image, labels)
            text_loss  = F.cross_entropy(logits_per_text, labels)

            loss = (image_loss + text_loss) / 2

            if config.n_gpu > 1: 
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()

            global_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                global_step += 1
                optimizer.step() # PYTORCH 1.x : call optimizer.step() first then scheduler.step()
                
                # logit scaling set as max 100 as mentioned in CLIP paper # log(100) = 4.6052
                if config.n_gpu == 1:
                    model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
                elif config.n_gpu > 1:
                    model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)

                if scheduler:
                    scheduler.step() 
                    
                model.zero_grad()

                if global_step % config.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f})".format(epoch, global_step, 
                        optimizer.param_groups[0]["lr"], loss.item(), global_loss / global_step)
                    )

                if (config.save_steps > 0 and global_step % config.save_steps == 0) or \
                        global_step == t_total:
                    # saving checkpoint
                    save_checkpoint(config, epoch, global_step, model, optimizer) 
                    

    return global_step, global_loss / global_step

def save_checkpoint(config, epoch, global_step, model, optimizer):
    '''
    Checkpointing. Saves model and optimizer state_dict() and current epoch and global training steps.
    '''
    checkpoint_path = os.path.join(config.saved_checkpoints, f'checkpoint_{epoch}_{global_step}.pth')
    save_num = 0
    while (save_num < 10):
        try:

            if config.n_gpu > 1:
                torch.save({
                    'epoch' : epoch,
                    'global_step' : global_step,
                    'model_state_dict' : model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)
            else:
                torch.save({
                    'epoch' : epoch,
                    'global_step' : global_step,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)

            logger.info("Save checkpoint to {}".format(checkpoint_path))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return


if __name__ == "__main__":
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.n_gpu = torch.cuda.device_count() # config.n_gpu 
    set_seed(seed=11, n_gpu=config.n_gpu)

    # getting text tokenizer
    tokenizer = TRIE_TOKENIZER('./ringrwkv/rwkv_vocab_v20230424.txt')

    # creating RN50 CLIP model
    model_params = dict(config.MixDLIP)
    model_params['vision_layers'] = tuple(model_params['vision_layers'])
    model_params['vision_patch_size'] = None
    model = DLIP(**model_params)

    logger.info(f"Training/evaluation parameters {train_config}")

    # getting dataset for training, normol: 5K  tiny: 100
    train_dataset = CIFAR100(root="./data", download=False, train=False)
    #train_dataset = CIFAR10(root="./data", download=False, train=False)

    # Now training
    global_step, avg_loss = train(config, train_dataset, model)
    
    logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)

