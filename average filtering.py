import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
 
img = cv.imread("c:/Users/arunk/Downloads/mvtec_anomaly_detection/carpet/test/cut/000.png")
assert img is not None, "file could not be read, check with os.path.exists()"
 
kernel = np.ones((3,3),np.float32)/9

dst = cv.filter2D(img,-1,kernel)

gaussian_blur = cv.GaussianBlur(img, (9, 9), 0)

cv.imshow('Original Image',img)
cv.imshow('Gaussian Kernel', gaussian_blur)
cv.imshow('Simple Averaging', dst)
cv.waitKey(0)
cv.destroyAllWindows()