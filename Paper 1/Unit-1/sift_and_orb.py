import cv2
import matplotlib.pyplot as plt

# Load the image (replace with your image path)
image = cv2.imread('C:/Users/arunk/Downloads/mvtec_anomaly_detection/capsule/test/crack/001.png', cv2.IMREAD_GRAYSCALE)

# SIFT feature extraction
sift = cv2.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(image, None)
sift_img = cv2.drawKeypoints(image, kp_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# ORB feature extraction
orb = cv2.ORB_create()
kp_orb, des_orb = orb.detectAndCompute(image, None)
orb_img = cv2.drawKeypoints(image, kp_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('SIFT Features')
plt.imshow(sift_img, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('ORB Features')
plt.imshow(orb_img, cmap='gray')
plt.show()  