import cv2
import numpy as np
import matplotlib.pyplot as plt

binary_mask = cv2.imread('C:/Users/arunk/Downloads/mvtec_anomaly_detection/metal_nut/ground_truth/bent/000_mask.png', cv2.IMREAD_GRAYSCALE)
_, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5, 5), np.uint8)

opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Original Binary Mask')
plt.imshow(binary_mask, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Opening')
plt.imshow(opening, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Closing')
plt.imshow(closing, cmap='gray')
plt.show()