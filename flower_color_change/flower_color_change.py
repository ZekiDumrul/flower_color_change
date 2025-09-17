import cv2
import numpy as np

image_path = r"C:\Users\Zeki\Desktop\resim\gul.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Görsel yüklenemedi! Tekrar deneyiniz")
    exit()

# HSV donusturme
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Kırmızı renk aralıgı
lower_red1 = np.array([0, 30, 30])  
upper_red1 = np.array([15, 255, 255])

lower_red2 = np.array([155, 30, 30])  
upper_red2 = np.array([180, 255, 255])

# Maskeler
mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)

# Maskeleri birleştir
mask = cv2.bitwise_or(mask1, mask2)

# Maske için morfolojik işlem
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel , iterations=6)  
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel , iterations=2)  

# Maskeyi yumuşatmak için Gaussian blur
mask = cv2.GaussianBlur(mask, (7, 7), 0)
_, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)

# Sadece kırmızı bölgeler
rose_only = cv2.bitwise_and(image, image, mask=mask)

# Mor efekti ile siyah arka plan
purple_rose_alt = rose_only.copy()
# Kırmızı kanalı azalt, mavi kanalı artır 
purple_rose_alt[:, :, 2] = purple_rose_alt[:, :, 2] * 0.3  
purple_rose_alt[:, :, 0] = purple_rose_alt[:, :, 0] * 1.5  
purple_rose_alt = np.clip(purple_rose_alt, 0, 255).astype(np.uint8)

# Siyah arka plan oluştur
black_background = np.zeros_like(image)


# Sonuç
final_result_alt = black_background.copy()
final_result_alt[mask > 0] = purple_rose_alt[mask > 0]

# Sonuçları göster
cv2.imshow("Original Image", image)
cv2.imshow("Mask", mask)
cv2.imshow("Purple Rose", final_result_alt)

cv2.waitKey(0)
cv2.destroyAllWindows()
