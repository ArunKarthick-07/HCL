import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('c:/Users/arunk/Downloads/mvtec_anomaly_detection/carpet/test/cut/000.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)  # Assuming defects are darker

kernel = np.ones((5, 5), np.uint8)

opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Binary Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Opening')
plt.imshow(opening, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Closing (Isolated Defects)')
plt.imshow(closing, cmap='gray')
plt.show()