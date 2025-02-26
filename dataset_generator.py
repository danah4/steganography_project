import cv2
import numpy as np
import os
from steg import embed_message  # استدعاء كود الإخفاء الذي كتبناه سابقًا

# إنشاء مجلدات لحفظ الصور
os.makedirs("dataset/clean", exist_ok=True)
os.makedirs("dataset/stego", exist_ok=True)

# عدد الصور لكل فئة
num_images = 100  
image_size = (256, 256)

for i in range(num_images):
    # إنشاء صورة عشوائية
    clean_image = np.random.randint(0, 256, (*image_size, 3), dtype=np.uint8)
    cv2.imwrite(f"dataset/clean/image_{i}.png", clean_image)

    # إنشاء صورة مخفية
    secret_message = "HiddenData"  # يمكن تغييرها حسب الحاجة
    stego_image_path = f"dataset/stego/image_{i}.png"
    cv2.imwrite("temp.png", clean_image)
    embed_message("temp.png", secret_message, stego_image_path)

print("✅ تم إنشاء مجموعة البيانات بنجاح!")
