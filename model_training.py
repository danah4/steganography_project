import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ✅ تأكدي من إنشاء المجلد قبل حفظ النموذج
model_dir = "model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# تحميل البيانات المستخرجة مسبقًا
data = np.load("dataset/features.npz")
X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

# تدريب نموذج Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# اختبار النموذج
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ✅ الآن يمكن حفظ النموذج بدون مشاكل
joblib.dump(model, os.path.join(model_dir, "random_forest_model.pkl"))

print(f"🎯 دقة النموذج: {accuracy * 100:.2f}%")
print(f"✅ تم حفظ النموذج المدرب في {model_dir}/random_forest_model.pkl")
