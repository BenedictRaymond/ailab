# Experiment 4: Image Processing Operations Using OpenCV in Python

Ex.No: 4 Perform image processing operations such as converting to grayscale, blurring, and
        edge detection using OpenCV in Python.

Date:

Aim:

           The aim of this experiment is to perform basic image processing operations such as converting
           an image to grayscale, applying blurring, and detecting edges using OpenCV in Python.

Procedure:

Step1: Import OpenCV, NumPy, and Matplotlib for image processing and display.

Step2: cv2.imread() to read the input image.

Step3: cv2.cvtColor() to convert the image from BGR to grayscale.

Step4: cv2.GaussianBlur() to apply a Gaussian blur to the grayscale image.

Step5: cv2.Canny() to detect edges in the blurred image.

Step6: Display the original, grayscale, blurred, and edge-detected images using cv2.imshow() or
matplotlib.

## Program

```python
pip3 install opencv-python

pip3 install matplotlib

#Import required packages

import cv2

import numpy as np

import matplotlib.pyplot as plt

img_color = cv2.imread('cat.jpg')

if img_color is None:

  print("Error loading image")

  exit()

In [5]:

#Converting the image to Grayscale

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

In [6]:

#Applying Gaussian Blur

kernel_size = 5 # Choose an odd integer value (e.g., 3, 5, 7)

img_blur = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)

In [7]:

#Perform Canny Edge detection

threshold1 = 50

threshold2 = 150

edges = cv2.Canny(img_blur, threshold1, threshold2)

#Display the processed images using matplotlib

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Original image (RGB)

axs[0, 0].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))

axs[0, 0].set_title('Original Image')

axs[0, 0].axis('off')

# Grayscale image

axs[0, 1].imshow(img_gray, cmap='gray')

axs[0, 1].set_title('Grayscale Image')

axs[0, 1].axis('off')

# Blurred image

axs[1, 0].imshow(img_blur, cmap='gray')

axs[1, 0].set_title(f'Blurred Image (Kernel {kernel_size}x{kernel_size})')

axs[1, 0].axis('off')

# Edge detection image

axs[1, 1].imshow(edges, cmap='gray')

axs[1, 1].set_title(f'Edge Detection ({threshold1}-{threshold2})')

axs[1, 1].axis('off')

plt.show()
```

## Output and Result

Output

Result:

Thus, the Python program to perform image processing operations such as converting to grayscale,
blurring, and edge detection using OpenCV was verified successfully.
