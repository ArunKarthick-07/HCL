import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the automotive part image (replace with your image path)
image = cv2.imread('c:/Users/arunk/Downloads/mvtec_anomaly_detection/hazelnut/test/crack/001.png', cv2.IMREAD_GRAYSCALE)

# Gaussian blur for smoothing
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Sharpening using Laplacian
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
sharpened = cv2.convertScaleAbs(image - laplacian)  # Subtract to highlight edges like scratches

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Gaussian Blurred')
plt.imshow(blurred, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Sharpened (Highlights Scratches/Dents)')
plt.imshow(sharpened, cmap='gray')
plt.show()