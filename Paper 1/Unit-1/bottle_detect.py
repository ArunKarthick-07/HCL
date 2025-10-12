import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the glass bottle image (replace with your image path)
image = cv2.imread('C:/Users/arunk/Downloads/mvtec_anomaly_detection/bottle/test/broken_large/001.png', cv2.IMREAD_GRAYSCALE)

# Edge detection (Canny)
edges = cv2.Canny(image, 50, 150)

# Morphology to enhance cracks/missing parts
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)
closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

# Find contours for localization
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for contour in contours:
    if cv2.contourArea(contour) > 50:  # Filter small noise
        cv2.drawContours(output, [contour], -1, (0, 0, 255), 2)  # Red for defects

# Display pipeline results
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.title('Original')
plt.imshow(image, cmap='gray')
plt.subplot(1, 4, 2)
plt.title('Canny Edges')
plt.imshow(edges, cmap='gray')
plt.subplot(1, 4, 3)
plt.title('After Morphology')
plt.imshow(closed, cmap='gray')
plt.subplot(1, 4, 4)
plt.title('Defect Localization')
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.show()