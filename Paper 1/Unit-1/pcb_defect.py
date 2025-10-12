import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the PCB image (replace with your image path)
image = cv2.imread('01_missing_hole_01.jpg', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)  # Assuming solder joints are bright

# Connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# Classify based on area (e.g., good: 50-200, defective: else)
min_good_area = 100
max_good_area = 700
good_count = 0
defective_count = 0

output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    if min_good_area <= area <= max_good_area:
        good_count += 1
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for good
    else:
        defective_count += 1
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for defective

# Statistics
total = good_count + defective_count
print(f"Statistics: Good: {good_count} ({good_count/total*100:.2f}%), Defective: {defective_count} ({defective_count/total*100:.2f}%), Total: {total}")

# Display
plt.figure(figsize=(10, 5))
plt.title('Solder Joints (Green: Good, Red: Defective)')
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.show()