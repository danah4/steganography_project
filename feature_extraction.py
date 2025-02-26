import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split

def load_images(folder):
    images = []
    labels = []
    
    for img_path in glob.glob(f"{folder}/*.png"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # تحويل الصورة إلى رمادي
        img = cv2.resize(img, (128, 128))  # تصغير الصورة لحجم موحد
        images.append(img.flatten())  # تحويل الصورة إلى مصفوفة رقمية
        labels.append(1 if "stego" in folder else 0)  # 1 = صورة مخفية، 0 = صورة نظيفة

    return np.array(images), np.array(labels)

# تحميل البيانات من المجلدات
X_clean, y_clean = load_images("dataset/clean")
X_stego, y_stego = load_images("dataset/stego")

# دمج البيانات
X = np.vstack((X_clean, X_stego))
y = np.hstack((y_clean, y_stego))

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# حفظ البيانات لاستخدامها في تدريب النموذج
np.savez("dataset/features.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

print(f"✅ تم استخراج الميزات بنجاح!")
print(f"📊 عدد بيانات التدريب: {len(X_train)}, عدد بيانات الاختبار: {len(X_test)}")
