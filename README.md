# Training and Inference for Integer-based Semantic Segmentation Network

### Overview
This repository contains an implementation of Training and Inference for Integer-based Semantic Segmentation Network along with pre-trained models.

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
2. [About EfficientNet-PyTorch](#about-efficientnet-pytorch)
4. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Classify](#example-classification)
    * [Example: Extract features](#example-feature-extraction)
    * [Example: Export to ONNX](#example-export)
6. [Contributing](#contributing)

### Installation

Install via pip:
```bash
pip install efficientnet_pytorch
```

Or install from source:
```bash
git clone https://github.com/lukemelas/EfficientNet-PyTorch
cd EfficientNet-Pytorch
pip install -e .
```

### Usage

#### Loading pretrained models

Load an EfficientNet:  
```python
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_name('efficientnet-b0')
```

Load a pretrained EfficientNet:
```python
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')
```

Note that pretrained models have only been released for `N=0,1,2,3,4,5` at the current time, so `.from_pretrained` only supports `'efficientnet-b{N}'` for `N=0,1,2,3,4,5`.

Details about the models are below:

|    *Name*         |*# Params*|*Top-1 Acc.*|*Pretrained?*|
|:-----------------:|:--------:|:----------:|:-----------:|
| `efficientnet-b0` |   5.3M   |    76.3    |      ✓      |
| `efficientnet-b1` |   7.8M   |    78.8    |      ✓      |
| `efficientnet-b2` |   9.2M   |    79.8    |      ✓      |
| `efficientnet-b3` |    12M   |    81.1    |      ✓      |
| `efficientnet-b4` |    19M   |    82.6    |      ✓      |
| `efficientnet-b5` |    30M   |    83.3    |      ✓      |
| `efficientnet-b6` |    43M   |    84.0    |      ✓      |
| `efficientnet-b7` |    66M   |    84.4    |      ✓      |


#### Example: Classification

Below is a simple, complete example. It may also be found as a jupyter notebook in `examples/simple` or as a [Colab Notebook](https://colab.research.google.com/drive/1Jw28xZ1NJq4Cja4jLe6tJ6_F5lCzElb4).

We assume that in your current directory, there is a `img.jpg` file and a `labels_map.txt` file (ImageNet class names). These are both included in `examples/simple`.

```python
import json
from PIL import Image
import torch
from torchvision import transforms

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')

# Preprocess image
tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
img = tfms(Image.open('img.jpg')).unsqueeze(0)
print(img.shape) # torch.Size([1, 3, 224, 224])

# Load ImageNet class names
labels_map = json.load(open('labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify
model.eval()
with torch.no_grad():
    outputs = model(img)

# Print predictions
print('-----')
for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))
```

#### Example: Feature Extraction

You can easily extract features with `model.extract_features`:
```python
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')

# ... image preprocessing as in the classification example ...
print(img.shape) # torch.Size([1, 3, 224, 224])

features = model.extract_features(img)
print(features.shape) # torch.Size([1, 1280, 7, 7])
```

#### Example: Export to ONNX  

Exporting to ONNX for deploying to production is now simple:
```python
import torch
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b1')
dummy_input = torch.randn(10, 3, 240, 240)

torch.onnx.export(model, dummy_input, "test-b1.onnx", verbose=True)
```

[Here](https://colab.research.google.com/drive/1rOAEXeXHaA8uo3aG2YcFDHItlRJMV0VP) is a Colab example.


#### ImageNet

See `examples/imagenet` for details about evaluating on ImageNet.

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   
