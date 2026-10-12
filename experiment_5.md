# Experiment 5: Video Processing Operations Using OpenCV

Ex.No:5        Perform video processing operations using OpenCV and apply simple filters or
               transformations to each frame of the video stream.

Date       :

Aim:

To perform video processing operations using OpenCV and apply simple filters or transformations to
each frame of the video stream, such as grayscale conversion, blurring, and edge detection, in real-
time.

Procedure:

Step1: Initialize Video Capture

Step2: Read Frames in a Loop.

Step3: Apply Image Processing Operations.

Step4: Display the Processed Frame

Step5: Exit the Video Stream

## Program

```python
import cv2

import numpy as np

In [28]:

# Function to apply various transformations to the frame

def transform_frame(frame):

  # Convert the frame to HSV (Hue, Saturation, Value) color space

  hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # Apply a color transformation by adjusting the Hue channel

  # Here we shift the hue slightly towards blue

  hsv_frame[:, :, 0] += 15 # Increase Hue by 15

  # Convert back to BGR for display

  transformed_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)

  # Apply a Gaussian blur to the frame

  blurred_frame = cv2.GaussianBlur(transformed_frame, (15, 15), 0)

  # Detect edges in the frame using the Canny edge detector

  edge_frame = cv2.Canny(blurred_frame, threshold1=30, threshold2=150)

  return edge_frame #Change this to visualize each effects

In [29]:

cap = cv2.VideoCapture('Sample.mp4')

if not cap.isOpened():

  print("Error: Could not open video.")

  exit()

In [30]:

frame_rate = 30 # Desired frame rate in frames per second

delay = int(1000 / frame_rate) # Convert frame rate to delay time in milliseconds

while True:

  # Read the next frame from the video

  ret, frame = cap.read()

  # Check if the frame was successfully read; if not, break the loop

  if not ret:

     print("Video Ends")

     break

   try:

     # Process the frame using a defined transform_frame function

     processed_frame = transform_frame(frame) # Replace with your actual processing logic

    # Display the processed frame in a window named 'Processed Frame'

    cv2.imshow('Processed Frame', processed_frame)

           except Exception as e:

     print(f"Error during frame transformation: {e}")

     break

   key = cv2.waitKey(delay) & 0xFF

  if key == ord('q'):

     break

Video Ends

In [31]:

cap.release()

cv2.destroyAllWindows()
```

## Output and Result

Output:

Result:

Thus, the Python program to perform video processing operations using OpenCV and apply simple
filters or transformations to each frame of the video stream was verified successfully.
