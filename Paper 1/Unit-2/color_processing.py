import cv2
import matplotlib.pyplot as plt
import numpy as np

# download the image
# https://www.mathworks.com/help/images/peppers.png

# Upload the image in your Files then read

img = cv2.imread('c:/Users/arunk/Downloads/peppers.png')
# Convert BGR to RGB for Matplotlib display

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Split the image into its R, G, B channels
b, g, r = cv2.split(img)

# Create a figure with subplots
plt.figure(figsize=(12, 6))

# Original RGB image
plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title('Original RGB Image')
plt.axis('off')

# Red Channel
plt.subplot(2, 2, 2)
plt.imshow(r, cmap='gray')
plt.title('Red Channel')
plt.axis('off')


# Green Channel
plt.subplot(2, 2, 3)
plt.imshow(g, cmap='gray')
plt.title('Green Channel')
plt.axis('off')

# Blue Channel
plt.subplot(2, 2, 4)
plt.imshow(b, cmap='gray')
plt.title('Blue Channel')
plt.axis('off')

plt.tight_layout()
plt.show()

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
# Split the HSV image into H, S, V channels
h, s, v = cv2.split(img_hsv)
 
# Create a figure with subplots
plt.figure(figsize=(12, 6))
 
# Original RGB image (for comparison)
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original RGB Image')
plt.axis('off')
 
# Hue Channel
plt.subplot(2, 2, 2)
plt.imshow(v, cmap='hsv') # Hue is often visualized with an HSV colormap
plt.title('Hue Channel')
plt.axis('off')


# Saturation Channel
plt.subplot(2, 2, 3)
plt.imshow(s, cmap='gray')
plt.title('Saturation Channel')
plt.axis('off')

# Value Channel
plt.subplot(2, 2, 4)
plt.imshow(v, cmap='gray')
plt.title('Value Channel')
plt.axis('off')

plt.tight_layout()
plt.show()