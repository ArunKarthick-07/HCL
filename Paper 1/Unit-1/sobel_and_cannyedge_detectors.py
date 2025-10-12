import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('C:/Users/arunk/Downloads/mvtec_anomaly_detection/metal_nut/train/good/219.png', cv2.IMREAD_GRAYSCALE)

sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.sqrt(sobel_x**2 + sobel_y**2)
sobel = cv2.convertScaleAbs(sobel)


canny = cv2.Canny(image, 100, 200)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Sobel')
plt.imshow(sobel, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Canny')
plt.imshow(canny, cmap='gray')
plt.show()