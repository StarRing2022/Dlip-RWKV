import torch
import torch.nn.functional as F
import numpy as np
import os
from omegaconf import OmegaConf

from dataset import Flickr30k
from dataloaders import get_dataloader

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
from tqdm import tqdm

from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

TRAINER_CONFIG_PATH = 'config/train_config.yaml'
MODEL_CONFIG_PATH = 'config/model_config.yaml'

train_config = load_config_file(TRAINER_CONFIG_PATH)
model_config = load_config_file(MODEL_CONFIG_PATH)

config = OmegaConf.merge(train_config, model_config)
logger = setup_logger("DLIP_Flickr30k_INFER", config.logs, 0, filename = "training_logs.txt")


def tokenize(config,text,tokenizer):
    input_ids = tokenizer.encode(text)
    context_length = 128 #与model_config中一致
    result = torch.zeros(context_length, dtype=torch.long)
    result[:len(input_ids)] = torch.tensor(input_ids)
    return result

def zeroshot_classifier(model, classnames, templates, tokenizer, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = templates + classname #format with class
            texts = tokenize(config,texts, tokenizer).to(device) #tokenize
            texts = texts.unsqueeze(0)
            #logger.info(texts.shape)
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def predict_class(model, images, image_names, dataset_classes, tokenizer, device):
    '''
    Classifies images by predicting their classes from "dataset_classes"
    '''
    with torch.no_grad():
        
        classnames = [classname for classname in dataset_classes]
        
        templates = "这副图片描述了，一个"
        zeroshot_weights = zeroshot_classifier(model, classnames, templates, tokenizer, device)
        # print("zeroshot_weights.shape", zeroshot_weights.shape)
        predictions = []

        for image, image_name in zip(images, image_names):
            image_input = image.to(device)
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity_scale = 35.0
            similarity = (similarity_scale * image_features @ zeroshot_weights).softmax(dim=-1)
            
            # top 5 predictions
            values, indices = similarity[0].cpu().topk(5)
            # print("------------------------")
            # print("img : ", image_name)
            # print("predicted classes :")
            for value, index in zip(values, indices):
                print(f"{classnames[index]:>16s}: {100 * value.item():.2f}%")
            # print("------------------------")   

            predictions.append((values, indices))  

            classname = classnames[indices[0]]
    
    return predictions, classname   


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

    # loading trained weights
    checkpoint = torch.load("./checkpoints/checkpoint_nollm_29_37500.pth")
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model = model.to(config.device)
    model.eval()

    # cifar100 = CIFAR100(root="./data", download=False, train=False)

    cifar10 = CIFAR10(root="./data", download=False, train=False)

    # print(cifar100.classes)
    # print(cifar10.classes)

    dataset_classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    
    transform = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)),
    ])

    transform_no_norm = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
    ])

    
    images = []
    image_names = []
    raw_images = []

    img_path = "apple.jpg"
    image_name = os.path.split(img_path)[-1]
    image = transform(Image.open(img_path)).unsqueeze(0)
    raw_image = transform_no_norm(Image.open(img_path))        
    raw_images.append(raw_image)
    images.append(image)
    image_names.append(image_name)

    #print(images,image_names,raw_images)

    predictions,classname = predict_class(model, images, image_names, dataset_classes, tokenizer, config.device)
    
    #print(classname)

    model = RwkvForCausalLM.from_pretrained("RWKV-4-World-0.4B")
    template = "这副图片描述了，一个"
    text = f"下面是一段关于一幅图片的描述，你作为一位知名的艺术专家，请更生动地描绘这张图片，字数在50字左右。原描述："+template+classname
    question = f'Question: {text.strip()}\n\nAnswer:'
    input_ids = tokenizer.encode(question)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    out = model.generate(input_ids,max_new_tokens=50)

    #print(out[0])

    outlist = out[0].tolist()

    for i  in outlist:
        if i==0:
            outlist.remove(i)

    #print(outlist)
    answer = tokenizer.decode(outlist)
    print(answer)