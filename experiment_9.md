# Experiment 9: MobileNetV2 Quantization Using FBGEMM and PyTorch

Ex.No:9         Perform Quantization on MobileNet V2 Model using fbgemm quantization
                method using Pytorch framework and evaluate.
Date    :

Aim:

To perform Quantization on MobileNet V2 Model using fbgemm quantization method using Pytorch
framework and evaluate.Components

Procedure:

 Step 1: Load the pretrained MobileNetV2 model.

  Step 2: Set up the model for quantization by fusing layers and preparing it for quantization-aware
training using FBGEMM.

 Step 3: Fine-tune the model (optional) to mitigate the accuracy loss due to quantization.

 Step 4: Convert the model to a quantized version.

 Step 5: Evaluate the quantized model’s performance.

## Program

```python
import torch

print(torch.backends.quantized.supported_engines) #Check for fpgemm

import torch

from torch import nn

from torchvision.models import mobilenet_v2

from torchvision import datasets, transforms

from torch.utils.data import DataLoader

# Step 1: Load a pre-trained MobileNetV2 model

model_fp32 = mobilenet_v2(pretrained=True)

model_fp32.eval()

# Step 2: Prepare the data (e.g., ImageNet or a similar dataset)

transform = transforms.Compose([

  transforms.Resize(256),

  transforms.CenterCrop(224),

  transforms.ToTensor(),

  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

# Using a sample dataset (CIFAR-10 in this case, but replace with your dataset)

dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Step 3: Fuse appropriate layers for MobileNetV2

# MobileNetV2 has depthwise separable convolutions, so we only need to fuse the first conv and batch
norm layers

model_fp32_fused = model_fp32

for module_name, module in model_fp32_fused.named_children():

  if module_name == "features":

     for idx, m in module.named_children():

       if isinstance(m, nn.Conv2d):

          # Fuse Conv2d + BatchNorm2d + ReLU in each block (if applicable)

         torch.quantization.fuse_modules(model_fp32_fused.features[int(idx)], ['0', '1', '2'],
inplace=True)

# Step 4: Specify quantization configuration

# Use fbgemm backend for x86 architectures

model_fp32_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Step 5: Prepare the model for static quantization

# This inserts observers in the model that will collect statistics

model_int8_prepared = torch.quantization.prepare(model_fp32_fused)

# Step 6: Calibrate the model by running inference

# Here, we pass the data through the model so that it can learn activation ranges

with torch.no_grad():

  for images, _ in dataloader:

     model_int8_prepared(images)

# Step 7: Convert to a quantized model

model_int8 = torch.quantization.convert(model_int8_prepared)

# Step 8: Evaluate the quantized model

def evaluate(model, dataloader):

  correct = 0

  total = 0

  model.eval()

  with torch.no_grad():

    for images, labels in dataloader:

          outputs = model(images)

          _, predicted = torch.max(outputs, 1)

          total += labels.size(0)

          correct += (predicted == labels).sum().item()

  print(f'Accuracy of the model: {100 * correct / total:.2f}%')

# Evaluate both the original FP32 model and the quantized model

print("Evaluating FP32 Model...")

evaluate(model_fp32, dataloader)

print("Evaluating INT8 Quantized Model...")

evaluate(model_int8, dataloader)
```

## Output and Result

Output:

Result:

Thus, the Python program to perform Quantization on MobileNet V2 Model using fbgemm
quantization method using Pytorch framework and evaluate.Components was verified successfully.
