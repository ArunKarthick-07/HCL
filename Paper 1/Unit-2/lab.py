
import cv2
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt

image = cv2.imread('c:/Users/arunk/Downloads/peppers.png')


# Convert RGB image to L*a*b*
# For scikit-image:
lab_image = color.rgb2lab(image)

# For OpenCV:
# BGR is the default for cv2.imread, so convert BGR to RGB first
if image.shape[-1] == 3: # Check if it's a color image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
else: # Grayscale image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
lab_image_cv2 = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

# Using scikit-image and matplotlib
l_channel = lab_image[:,:,0]
a_channel = lab_image[:,:,1]
b_channel = lab_image[:,:,2]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))


# Display the L* channel
axes[0].imshow(l_channel, cmap='gray')
axes[0].set_title('L* Channel (Lightness)')
axes[0].axis('off')

# Display the a* channel
axes[1].imshow(a_channel, cmap='gray')
axes[1].set_title('a* Channel (Green-Red)')
axes[1].axis('off')

# Display the b* channel
axes[2].imshow(b_channel, cmap='gray')
axes[2].set_title('b* Channel (Blue-Yellow)')
axes[2].axis('off')

plt.tight_layout()
plt.show()
