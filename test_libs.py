# Import libraries
import tensorflow as tf
import cv2

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Test OpenCV
image = cv2.imread('./test.jpg')
print("Image shape:", image.shape)
