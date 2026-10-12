# Experiment 10: L1 Structured Pruning on ResNet18 Using PyTorch

Ex.No:10              Perform L1 structured Pruning on ResNet18 model using pytorch
                      framework and evaluate.

Date      :
Aim:

.    To perform L1 structured Pruning on ResNet18 model using pytorch framework and evaluate.
Procedure:

Step1: Load Pre-trained ResNet18

Step2: Define L1 Pruning on Convolutional Layers

Step3: Remove Pruning Masks

Step4: Evaluate the Pruned Model

Step5: Fine-tune the Pruned Model

## Program

```python
import torch

import torch.nn as nn

import torch.nn.utils.prune as prune

import torchvision.models as models

import torchvision.transforms as transforms

import torchvision.datasets as datasets

from torch.utils.data import DataLoader

import torch.optim as optim

import torch.nn.functional as F

In [2]:

# 1. Load the pre-trained ResNet-18 model

model = models.resnet18(pretrained=True)

# 2. Function to apply L1 structured pruning

def l1_structured_pruning(model, amount):

  for name, module in model.named_modules():

       if isinstance(module, nn.Conv2d):

          # Apply L1-norm structured pruning on Conv2d layers

          prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)

In [4]:

# 3. Prune 30% of filters in the convolutional layers

l1_structured_pruning(model, amount=0.3)

In [5]:

# 4. Prepare the CIFAR-10 dataset and DataLoader for evaluation

transform = transforms.Compose([

     transforms.Resize(224),

     transforms.ToTensor(),

     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

# Load CIFAR-10 validation dataset

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

In [7]:

# 5. Evaluation function to test the pruned model on CIFAR-10

def evaluate(model, dataloader):

     model.eval()

     correct = 0

     total = 0

     with torch.no_grad():

       for inputs, labels in dataloader:

          inputs, labels = inputs.cuda(), labels.cuda()

          outputs = model(inputs)

          _, predicted = torch.max(outputs.data, 1)

          total += labels.size(0)

          correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total

  return accuracy

# 6. Move the model to GPU for evaluation (if available)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

In [9]:

# 7. Evaluate the pruned model

accuracy = evaluate(model, test_loader)

print(f'Accuracy of the pruned ResNet-18 model on CIFAR-10 test images: {accuracy:.2f}%')
```

## Output and Result

output:

Result:

Thus the to perform L1 structured Pruning on ResNet18 model using pytorch framework and the output
was verified successfully.
