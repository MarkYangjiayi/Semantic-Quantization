# Training and Inference for Integer-based Semantic Segmentation Network

### Overview
This repository contains a Tensorflow implementation of Training and Inference for Integer-based Semantic Segmentation Network along with pre-trained models.  

The goal of this implementation is to help you understand the implementation of quantization framework for semantic segmentation, and integrade them to your own network if possible.

### Table of contents
<!-- 1. [About EfficientNet](#about-efficientnet)
2. [About EfficientNet-PyTorch](#about-efficientnet-pytorch)
3. [Installation](#installation)
4. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Classify](#example-classification)
    * [Example: Extract features](#example-feature-extraction)
    * [Example: Export to ONNX](#example-export)
6. [Contributing](#contributing)  -->
1. [Installation](#about-efficientnet)
2. [Download Data](#Download Data)
3. [Usage](#usage)
    * [Training](#loading-pretrained-models)
    * [Evaluation](#example-classification)
    * [Visualization](#example-feature-extraction)
3. [Contributing](#contributing)

### Installation
Make sure you have **python3** installed in your environment.  
Type the following commands:
```bash
git clone https://github.com/MarkYangjiayi/Semantic-Quantization
cd Semantic-Quantization
pip install -r requirements.txt
```

### Download Data
We use TFrecord to feed data into our network, the code references DeepLabv3+ from google, which you can find [here](https://github.com/tensorflow/models/tree/master/research/deeplab).<br/>
To train the model with PASCAL VOC 2012 dataset, first download [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and [SegmentationClassAug](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0). Do not unzip and move them into the "./data" folder. Then,
```bash
cd data
sh convert_voc2012_aug.sh
```
If successful, you should have an "./data/pascal_voc_seg/tfrecord" folder with the dataset ready in TFRecords format.
Reference this [blog post](https://www.sun11.me/blog/2018/how-to-use-10582-trainaug-images-on-DeeplabV3-code/) for more information.

### Usage
* Training
```python
mode = "train"
```
* Evaluation
```python
mode = "val"
```
* Visualization  
```python
mode = "vis"
```
Note that the output prediction and ground truth is labeled with single channel. To gain better visualization, you have to convert them into RGB images.

### Contributing
If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   
