# Experiment 6: Build an Image Classification Model and Convert to ONNX Using PyTorch

Ex.No:6              Build a Multi layer perceptron using Pytorch framework and train a model
                     to perform image classification, convert the model into ONNX (Open-Neural
                     Network) format.

Date      :

Aim:

To write a python code to build a Multi layer perceptron using Pytorch framework and train a model
to perform image classification, convert the model into ONNX (Open-Neural Network) format.

Procedure:

Step1: Data Preparation.

Step2: Define the MLP Model

Step3: Training the Model

Step4 Evaluate the Model

Step5: Convert the Model to ONNX Format

## Program

```python
import torch

import torch.nn as nn

import torch.optim as optim

import torchvision

import torchvision.transforms as transforms

import numpy as np

import matplotlib.pyplot as plt

In [3]:

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

In [4]:

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

In [5]:

train_dataset = torchvision.datasets.FashionMNIST(root='./data',                train=True,      download=True,
transform=transform)

test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True,

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)

In [7]:

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

In [58]:

def imshow(img):

  img = img / 2 + 0.5 #unnormalize

  npimg = img.numpy()

  plt.imshow(np.transpose(npimg, (1, 2, 0)))

  plt.show()

In [24]:

dataiter = iter(train_loader)

images, labels = next(dataiter)

In [25]:

imshow(torchvision.utils.make_grid(images))

print(' '.join('%5s' % classes[labels[j]] for j in range(8)))

class FashionCNN(nn.Module):

  def      init (self):

     super(FashionCNN, self). init ()

     self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

     self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

     self.pool = nn.MaxPool2d(2, 2)

     self.fc1 = nn.Linear(64 * 7 * 7, 128)

     self.fc2 = nn.Linear(128, 10)

     self.relu = nn.ReLU()

     self.dropout = nn.Dropout(0.25)

  def forward(self, x):

     x = self.pool(self.relu(self.conv1(x)))

     x = self.pool(self.relu(self.conv2(x)))

     x = x.view(-1, 64 * 7 * 7) #flatten the tensor

     x = self.relu(self.fc1(x))

     x = self.dropout(x)

     x = self.fc2(x)

     return x

In [15]:

model = FashionCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):

  running_loss = 0.0

  for i, (images, labels) in enumerate(train_loader):

     images, labels = images.to(device), labels.to(device)

     # Zero the parameter gradients

     optimizer.zero_grad()

     # Forward pass

     outputs = model(images)

     loss = criterion(outputs, labels)

     # Backward pass and optimize

     loss.backward()

     optimizer.step()

     running_loss += loss.item()

     if i % 100 == 99: # Print every 100 batches

       print(f'Epoch     [{epoch+1}/{num_epochs}],           Step     [{i+1}/{len(train_loader)}],   Loss:
{running_loss/100:.4f}')

        running_loss = 0.0

print('Finished Training')

dataiter = iter(test_loader)

images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(8)))

imshow(torchvision.utils.make_grid(images))

with torch.no_grad():

  images = images.to(device)

  outputs = model(images)

  _, predicted = torch.max(outputs, 1)

  print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(8)))

model.eval()

correct = 0

total = 0

with torch.no_grad():

  for images, labels in test_loader:

     images, labels = images.to(device), labels.to(device)

     outputs = model(images)

     _, predicted = torch.max(outputs, 1)

     total += labels.size(0)

     correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the 10,000 test images: {100 * correct / total:.2f}%')

import torch.onnx

dummy_input = torch.randn(1, 1, 28, 28).to(device) #dummy just for conversion

torch.onnx.export(model,

           dummy_input,

           "fashion_mnist.onnx",

           export_params=True,

           opset_version=11,

           do_constant_folding=True,

           input_names=['input'],

           output_names=['output'],

           dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}) #Dynamic batch sizes

print("Model has been successfully converted to ONNX format.")

import onnxruntime as ort

import numpy as np

# Load the ONNX model

ort_session = ort.InferenceSession("fashion_mnist.onnx")

# Function to preprocess input images (similar to PyTorch's transforms)

def preprocess(image):

  image = image / 2 + 0.5 # Unnormalize if necessary

  image = np.array(image) # Convert to NumPy array

  image = np.expand_dims(image, axis=0) # Add batch dimension

  return image.astype(np.float32)

# Get a batch of images and labels for inference

dataiter = iter(test_loader)

images, labels = next(dataiter)

# Preprocess the input (ONNX requires NumPy input)

images_np = preprocess(images[0]) #Adjust for different image than first

# Run the ONNX model

ort_inputs = {'input': images_np}

ort_outs = ort_session.run(None, ort_inputs)

predicted_class = np.argmax(ort_outs[0], axis=1)

print(predicted_class)

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

## Output and Result

Output:

Result:

Thus the python code to build a Multi layer perceptron using Pytorch framework and train a model to
perform image classification, convert the model into ONNX (Open-Neural Network) format was
verified successfully.
