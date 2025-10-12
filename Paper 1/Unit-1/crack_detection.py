import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('c:/Users/arunk/Downloads/mvtec_anomaly_detection/screw/test/scratch_head/000.png', cv2.IMREAD_GRAYSCALE)

laplacian = cv2.Laplacian(image,cv2.CV_64F,1)
laplacian = cv2.convertScaleAbs(laplacian)

gaussian_blur = cv2.GaussianBlur(image, (5, 5), 1)

high_pass = cv2.subtract(image, gaussian_blur)

plt.figure(figsize=(15, 10))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image,cmap='gray')
plt.axis('off')


plt.subplot(1, 3, 2)
plt.title('Crack Defects')
plt.imshow(high_pass, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Crack Defects')
plt.imshow(laplacian, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()