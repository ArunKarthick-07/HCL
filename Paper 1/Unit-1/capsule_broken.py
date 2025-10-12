import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('C:/Users/arunk/Downloads/mvtec_anomaly_detection/capsule/test/crack/001.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)  # Assuming objects are darker

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

expected_count = 10
detected_count = num_labels - 1  # Excluding background

# Calculate average area of detected tablets
areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
avg_area = np.mean(areas)

# Loop through components and mark defects
output = image.copy()
for i in range(1, num_labels):  # skip background (0)
    x, y, w, h, area = stats[i]

    defect_type = None
    if area < 3.5 * avg_area:  # Too small → Broken
        defect_type = "Broken"
        color = (0, 0, 255)  # Red
    elif area > 4.5 * avg_area:  # Too large → Extra
        defect_type = "Extra"
        color = (255, 0, 0)  # Blue
    else:
        defect_type = "OK"
        color = (0, 255, 0)  # Green

    cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    cv2.putText(output, defect_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, color, 2)

if detected_count < expected_count:
    print(f"Missing tablets: {expected_count - detected_count}")

# Show results
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Tablet Defect Detection")
plt.axis("off")
plt.show()