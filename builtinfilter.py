import cv2
import numpy as np

# Load image
image = cv2.imread('c:/Users/arunk/Downloads/mvtec_anomaly_detection/capsule/train/good/186.png')

# 1. Sobel Filter (Horizontal and Vertical Edge Detection)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

# 2. Canny Filter (Edge Detection)
canny_edges = cv2.Canny(image, 0, 100)

# 3. Laplacian Filter (Edge Detection)
laplacian = cv2.Laplacian(image, cv2.CV_64F)

# 4. Gaussian Blur Filter (Noise Reduction)
gaussian_blur = cv2.GaussianBlur(image, (9, 9), 0)

# 5. Median Blur Filter (Noise Reduction)
median_blur = cv2.medianBlur(image, 5)

# 6. Bilateral Filter (Edge-Preserving Smoothing)
bilateral = cv2.bilateralFilter(image, 9, 75, 75)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Sobel X', sobel_x)
cv2.imshow('Sobel Y', sobel_y)
cv2.imshow('Canny', canny_edges)
cv2.imshow('Laplacian', laplacian)
cv2.imshow('Gaussian Blur', gaussian_blur)
cv2.imshow('Median Blur', median_blur)
cv2.imshow('Bilateral', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Save results
cv2.imwrite('sobel_x.jpg', sobel_x)
cv2.imwrite('sobel_y.jpg', sobel_y)
cv2.imwrite('canny.jpg', canny_edges)
cv2.imwrite('laplacian.jpg', laplacian)
cv2.imwrite('gaussian_blur.jpg', gaussian_blur)
cv2.imwrite('median_blur.jpg', median_blur)
cv2.imwrite('bilateral.jpg', bilateral)