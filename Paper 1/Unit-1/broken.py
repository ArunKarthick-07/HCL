import cv2
import numpy as np
image = cv2.imread('c:/Users/arunk/Downloads/mvtec_anomaly_detection/grid/test/broken/001.png')
# 1. Sobel Filter (Horizontal and Vertical Edge Detection)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

# 2. Canny Filter (Edge Detection)
canny_edges = cv2.Canny(image, 0, 100)

# 3. Laplacian Filter (Edge Detection)
laplacian = cv2.Laplacian(image, cv2.CV_64F)

cv2.imshow('Original', image)
cv2.imshow('Sobel X', sobel_x)
cv2.imshow('Sobel Y', sobel_y)
cv2.imshow('Canny', canny_edges)
cv2.imshow('Laplacian', laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()