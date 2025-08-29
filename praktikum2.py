import cv2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# Baca gambar
img_rgb = cv2.imread("kocheng.jpg")

# Konversi ke grayscale
gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Konversi ke grayscale custom
gray_custom = np.dot(img_rgb[..., :3], [0.5, 0.3, 0.2]).astype(np.uint8)

# --- Visualisasi ---
plt.subplot(1, 3, 1)
plt.title("RGB")
plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Grayscale")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Grayscale Custom")
plt.imshow(gray_custom, cmap="gray")
plt.axis("off")

plt.show()

# --- Simpan data pixel ---
print("Menyimpan data pixel...")

# DataFrame
df_rgb = pd.DataFrame(img_rgb.reshape(-1, 3), columns=["R", "G", "B"])
df_gray = pd.DataFrame(gray.reshape(-1, 1), columns=["Gray"])

# Simpan CSV (full data, tanpa batas baris)
df_rgb.to_csv("pixel_rgb.csv", index=False)
df_gray.to_csv("pixel_gray.csv", index=False)

# Simpan Excel (hanya 100 ribu baris pertama biar aman)
with pd.ExcelWriter("pixel_values.xlsx") as writer:
    df_rgb.head(100000).to_excel(writer, sheet_name="RGB", index=False)
    df_gray.head(100000).to_excel(writer, sheet_name="Gray", index=False)

print("✅ Data pixel berhasil disimpan:")
print("- pixel_rgb.csv (semua data RGB)")
print("- pixel_gray.csv (semua data Grayscale)")
print("- pixel_values.xlsx (sample 100 ribu baris pertama)")
