import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('C:/Users/arunk/Downloads/mvtec_anomaly_detection/capsule/test/crack/001.png', cv2.IMREAD_GRAYSCALE)

_, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Otsu Thresholding')
plt.imshow(otsu, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Adaptive Thresholding')
plt.imshow(adaptive, cmap='gray')
plt.show()