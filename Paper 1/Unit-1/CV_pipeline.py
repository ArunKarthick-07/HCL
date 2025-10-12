import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images with error handling
image = cv2.imread('C:/Users/arunk/Downloads/mvtec_anomaly_detection/capsule/test/crack/001.png', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('C:/Users/arunk/Downloads/mvtec_anomaly_detection/capsule/test/good/001.png', cv2.IMREAD_GRAYSCALE)
if image is None or template is None:
    raise FileNotFoundError("Could not load one or both images")

# Edge detection
edges = cv2.Canny(image, 100, 200)

# Morphological cleaning
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Template matching
res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
image_with_rect = image.copy()
cv2.rectangle(image_with_rect, top_left, bottom_right, 255, 2)

# Defect map via differencing
diff = cv2.absdiff(image, template)
_, defect_map = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Visualization
plt.figure(figsize=(15, 5))
plt.subplot(1, 5, 1)
plt.title('Edges (Defect Localization)')
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.subplot(1, 5, 2)
plt.title('Cleaned Morphology')
plt.imshow(cleaned, cmap='gray')
plt.axis('off')
plt.subplot(1, 5, 3)
plt.title('Template')
plt.imshow(template, cmap='gray')
plt.axis('off')
plt.subplot(1, 5, 4)
plt.title('Pattern Matching')
plt.imshow(image_with_rect, cmap='gray')
plt.axis('off')
plt.subplot(1, 5, 5)
plt.title('Defect Map')
plt.imshow(defect_map, cmap='gray')
plt.axis('off')
plt.show()

