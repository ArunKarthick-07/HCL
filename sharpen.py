import cv2
import numpy as np

# Load image
image = cv2.imread('c:/Users/arunk/Downloads/mvtec_anomaly_detection/capsule/train/good/186.png')

# Downsample (reduce detail) - halve the resolution
scale_factor = 0.5
width = int(image.shape[1] * scale_factor)
height = int(image.shape[0] * scale_factor)
downsampled = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

# Sharpen the downsampled image
kernel = np.array([[1, 1, 1],
                   [1,  1, 1],
                   [1, 1, 1]])/25
sharpened = cv2.filter2D(downsampled, -1, kernel)

# Save or display the result
cv2.imwrite('output.jpg', sharpened)
cv2.imshow('Sharpened Image', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()