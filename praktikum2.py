import cv2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

img_rgb = cv2.imread("kocheng.jpg")

# konversi ke grayscale
gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# konversi ke grayscale dengan bobot : 
gray_custom = np.dot(img_rgb[...,:3], [0.5, 0.3, 0.2]).astype(np.uint8)

plt.subplot(1,3,1)
plt.title("RGB")
plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)) # konversi BGR ke RGB
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Grayscale")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Grayscale")
plt.imshow(gray_custom, cmap="gray")
plt.axis("off")

plt.show()