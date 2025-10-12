import cv2
import matplotlib.pyplot as plt

image = cv2.imread('c:/Users/arunk/Downloads/mvtec_anomaly_detection/metal_nut/test/scratch/001.png', cv2.IMREAD_GRAYSCALE)

_, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

_, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.title('Original')
plt.imshow(image, cmap='gray')
plt.subplot(1, 4, 2)
plt.title('Global Thresholding')
plt.imshow(global_thresh, cmap='gray')
plt.subplot(1, 4, 3)
plt.title('Adaptive Thresholding')
plt.imshow(adaptive, cmap='gray')
plt.subplot(1, 4, 4)
plt.title('Otsu Thresholding')
plt.imshow(otsu, cmap='gray')
plt.show()

# Comparison: Print a note
print("Comparative results: Global may over/under segment, Adaptive handles varying lighting, Otsu is optimal for bimodal histograms.")