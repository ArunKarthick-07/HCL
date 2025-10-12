import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def classify_fruit(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Example thresholds for ripe (e.g., red/yellow) vs unripe (green); adjust based on fruit type
    lower_ripe = np.array([0, 50, 50])    # Red-ish
    upper_ripe = np.array([30, 255, 255])
    lower_unripe = np.array([35, 50, 50]) # Green-ish
    upper_unripe = np.array([85, 255, 255])
    
    mask_ripe = cv2.inRange(hsv, lower_ripe, upper_ripe)
    mask_unripe = cv2.inRange(hsv, lower_unripe, upper_unripe)
    
    ripe_pixels = cv2.countNonZero(mask_ripe)
    unripe_pixels = cv2.countNonZero(mask_unripe)
    
    classification = 'Ripe' if ripe_pixels > unripe_pixels else 'Unripe'
    return classification, mask_ripe + mask_unripe  # Combined mask for visualization

# Test on 5 samples (replace with your image paths)
image_paths = ['57.jpg', '58.jpg', '6.jpg', '7.jpg', '8.jpg','9.jpg']
results = []

plt.figure(figsize=(15, 5))
for i, path in enumerate(image_paths):
    if os.path.exists(path):
        classification, mask = classify_fruit(path)
        results.append((path, classification))
        plt.subplot(2, 6, i+1)
        plt.title(f'{os.path.basename(path)}: {classification}')
        plt.imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
        plt.subplot(2, 6, i+7)
        plt.title('Mask')
        plt.imshow(mask, cmap='gray')
    else:
        print(f"File {path} not found.")

plt.show()
print("Classification results:", results)