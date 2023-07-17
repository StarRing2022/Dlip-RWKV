# Dlip-RWKV
一种基于Clip改进的，通用HF格式的冻结LLM语言模型进行图文对齐训练的方案，以RWKV-4-World-0.4B为例，Cifar10为数据集 <br>

共创合作：受到visualrwkv冻结LLM模型启发（https://github.com/howard-hou/VisualRWKV）<br>


1.工程介绍：<br>
为找到一种通用的（而非像LLAMA Adapter等只适合某一LLM模型）HF格式冻结LLM模型对图像进行编码的图文对齐方案，目前以RWKV-4-World-0.4B为冻结LLM模型，将此模型融入了Clip

2.主要代码说明：<br>
config文件夹为模型和训练的配置文件<br>
model/model.py：Clip+LLM融合编码模型<br>
dataset.py：构造cifar10数据集的格式与图像增强方法<br>
dataloaders.py：获取数据集loader<br>
train_cifar.py：训练Dlip-RWKV模型<br>
infer.py：Dlip-RWKV模型推理（即判别图片的类别，并进行文本描述）<br>

3.使用：<br>
环境：WIN10+Torch1.31+Cuda11.6<br>
ringrwkv库 见https://github.com/StarRing2022/RingRWKV <br>
python train_cifar.py<br>
python infer.py<br>
测试结果在result.png所示<br>
由于使用的是小型的cifar10数据集，像素也不是很高，因而准确率还有待提高

RWKV-4-World-0.4B模型及训练30个epoch后的checkpoint文件：<br>
HF开源地址：https://huggingface.co/StarRing2022/Dlip-RWKV/
