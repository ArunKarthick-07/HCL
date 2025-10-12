import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('c:/Users/arunk/Downloads/mvtec_anomaly_detection/metal_nut/test/scratch/001.png', cv2.IMREAD_GRAYSCALE)

sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.convertScaleAbs(np.sqrt(sobel_x**2 + sobel_y**2))

canny = cv2.Canny(image, 100, 200)

laplacian = cv2.Laplacian(image, cv2.CV_64F,3)
laplacian = cv2.convertScaleAbs(laplacian)

plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.title('Original')
plt.imshow(image, cmap='gray')
plt.subplot(1, 4, 2)
plt.title('Sobel')
plt.imshow(sobel, cmap='gray')
plt.subplot(1, 4, 3)
plt.title('Canny')
plt.imshow(canny, cmap='gray')
plt.subplot(1, 4, 4)
plt.title('Laplacian')
plt.imshow(laplacian, cmap='gray')
plt.show()
