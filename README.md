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
Make sure you have python3 installed in your environment.  
Type the following commands:
```bash
git clone https://github.com/MarkYangjiayi/Semantic-Quantization
cd Semantic-Quantization
pip install -r requirements.txt
```

### Download Data
We use TFrecord to feed data into our network, the code references DeepLabv3+ from google, which you can find [here]().<br/>
To train with the "trainaug" dataset, reference this [blog post](https://www.sun11.me/blog/2018/how-to-use-10582-trainaug-images-on-DeeplabV3-code/).

### Usage
* Training
```bash
python main.py --mode "train"
```
* Evaluation
```bash
python main.py --mode "val"
```
* Visualization  
```bash
python main.py --mode "vis"
```
Note that the output prediction and ground truth is labeled with single channel. To gain better visualization, use color_convert.py to convert them into RGB images.

### Contributing
If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   
