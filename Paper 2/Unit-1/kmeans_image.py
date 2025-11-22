import numpy as np
from skimage import io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

img = io.imread('kfc_product.jpg')  
print("Original shape:", img.shape)

pixels = img.reshape(-1, 3)

k = 4  
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(pixels)

segmented_pixels = kmeans.cluster_centers_.astype('uint8')[labels]

segmented_img = segmented_pixels.reshape(img.shape)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img)
plt.axis('off')

plt.subplot(1,2,2)
plt.title(f"Segmented (k={k})")
plt.imshow(segmented_img)
plt.axis('off')
plt.tight_layout()
plt.show()
