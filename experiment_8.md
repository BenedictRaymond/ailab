# Experiment 8: Transfer Learning with ResNet18 on CIFAR-10

Ex.No:8            Perform Transfer Learning using Pytorch framework on ResNet18 pertained
                   model to fit CIFAR-10 dataset and evaluate the model.

 Date     :

 Aim:

 To perform Transfer Learning using Pytorch framework on ResNet18 pertained model to fit CIFAR-
 10 dataset and evaluate the model.

 Procedure:

 Step1: Load Pretrained ResNet18 Model

 Step2: Prepare the CIFAR-10 Dataset

 Step3: Freeze Pretrained Layers (Optional)

 Step4: Train the Model

 Step5: Evaluate the Model

## Program

```python
pip install torch torchvision numpy matplotlib

 import torch

 import torch.nn as nn

 import torch.optim as optim

 import torchvision

 import torchvision.transforms as transforms

 from torchvision import models

 from torch.utils.data import DataLoader

 import numpy as np

 import matplotlib.pyplot as plt

 transform = transforms.Compose([

   transforms.Resize(224), # ResNet models expect 224x224 inputs

   transforms.RandomHorizontalFlip(),

   transforms.ToTensor(),

   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load training and test sets

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,

                                download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,

                               download=True, transform=transform)

test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',

          'dog', 'frog', 'horse', 'ship', 'truck')

# Load the pre-trained ResNet18 model

model = models.resnet18(pretrained=True)

# Freeze the model's parameters so they don't get updated during backprop

for param in model.parameters():

   param.requires_grad = False

# Replace the fully connected layer to match CIFAR-10 classes

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 10) # CIFAR-10 has 10 classes

In [7]:

criterion = nn.CrossEntropyLoss() # Loss function

optimizer = optim.Adam(model.fc.parameters(), lr=0.001) # Only updating the final layer

In [8]:

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

num_epochs = 5

for epoch in range(num_epochs):

  running_loss = 0.0

  model.train()

  for i, data in enumerate(train_loader, 0):

     inputs, labels = data

     inputs, labels = inputs.to(device), labels.to(device)

     # Zero the parameter gradients

     optimizer.zero_grad()

     # Forward + backward + optimize

     outputs = model(inputs)

     loss = criterion(outputs, labels)

     loss.backward()

     optimizer.step()

     running_loss += loss.item()

     if i % 100 == 99: # Print every 100 batches

          print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")

          running_loss = 0.0

print('Finished Training')

Evaluation

In [9]:

correct = 0

total = 0

model.eval()

with torch.no_grad():

  for data in test_loader:

     images, labels = data

     images, labels = images.to(device), labels.to(device)

     outputs = model(images)

     _, predicted = torch.max(outputs.data, 1)

     total += labels.size(0)

     correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

torch.save(model.state_dict(), 'resnet18_cifar10.pth')

Inference

In [12]:

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 10)

# Load the saved model weights

model.load_state_dict(torch.load('resnet18_cifar10.pth'))

# Set the model to evaluation mode

model.eval()

from PIL import Image

transform = transforms.Compose([

  transforms.Resize(224),

  transforms.ToTensor(),

  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])

# Load a sample image (assuming it's in PIL format)

img = Image.open('goldcat.jpg')

# Preprocess the image

img = transform(img)

# Add batch dimension (N=1)

img = img.unsqueeze(0)

In [15]:

# Check if GPU is available and move model to device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

# Move the image to the device

img = img.to(device)

# Perform inference

with torch.no_grad():

   outputs = model(img)

# Get the predicted class

_, predicted_class = torch.max(outputs, 1)

# Map the class index to the class name (CIFAR-10 classes)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(f'Predicted class: {classes[predicted_class.item()]}')
```

## Output and Result

Output:

Predicted class: cat

Result:

Thus, the Python program to perform Transfer Learning using Pytorch framework on ResNet18
pertained model to fit CIFAR-10 dataset and evaluate the model was verified successfully.
