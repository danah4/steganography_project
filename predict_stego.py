import cv2
import numpy as np
import joblib
import os

# تحميل النموذج المدرب
model_path = "model/random_forest_model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError("❌ النموذج غير موجود! تأكد من تشغيل model_training.py لتدريب النموذج.")

model = joblib.load(model_path)

def preprocess_image(image_path):
    """ تجهيز الصورة لاستخدامها في التنبؤ """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # تحويل إلى رمادي
    img = cv2.resize(img, (128, 128))  # تصغير الصورة لحجم موحد
    img = img.flatten().reshape(1, -1)  # تحويلها إلى شكل مناسب للنموذج
    return img

def predict_image(image_path):
    """ تحديد ما إذا كانت الصورة تحتوي على بيانات مخفية أم لا """
    img = preprocess_image(image_path)
    prediction = model.predict(img)[0]
    
    if prediction == 1:
        return "🔍 الصورة تحتوي على بيانات مخفية (Stego Image)"
    else:
        return "✅ الصورة نظيفة (Clean Image)"

# تجربة النموذج على صور جديدة
test_image_1 = "dataset/stego/image_5.png"
test_image_2 = "dataset/clean/image_3.png"

print(f"{test_image_1} -> {predict_image(test_image_1)}")
print(f"{test_image_2} -> {predict_image(test_image_2)}")
