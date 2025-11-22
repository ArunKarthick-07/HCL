import cv2
import matplotlib.pyplot as plt

image = cv2.imread(r'c:\Users\arunk\OneDrive\Documents\HCL\Paper 1\Unit-1\kfc_product.jpg', cv2.IMREAD_GRAYSCALE)
template = cv2.imread(r'c:\Users\arunk\OneDrive\Documents\HCL\Paper 1\Unit-1\KFC_logo.png', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(image, None)
kp2, des2 = orb.detectAndCompute(template, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
orb_match_img = cv2.drawMatches(image, kp1, template, kp2, matches[:10], None, flags=2)

res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
template_match_img = image.copy()
cv2.rectangle(template_match_img, top_left, bottom_right, 255, 2)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title('ORB Feature Matching')
plt.imshow(orb_match_img, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Template Matching')
plt.imshow(template_match_img, cmap='gray')
plt.show()

# Comparison note
print("Comparison: ORB is robust to scale/rotation, template matching is simpler but sensitive to variations.")