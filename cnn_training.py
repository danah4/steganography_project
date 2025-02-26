import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# تحميل البيانات المستخرجة مسبقًا
data = np.load("dataset/features.npz")
X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

# إعادة تشكيل البيانات لتناسب CNN
X_train = X_train.reshape(-1, 128, 128, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 128, 128, 1).astype("float32") / 255.0

# بناء نموذج CNN
cnn_model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# تجميع النموذج
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# تدريب النموذج
cnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# تقييم النموذج
test_loss, test_acc = cnn_model.evaluate(X_test, y_test)
print(f"🎯 دقة نموذج CNN: {test_acc * 100:.2f}%")

# حفظ النموذج المدرب
os.makedirs("model", exist_ok=True)
cnn_model.save("model/cnn_model.h5")
print("✅ تم حفظ النموذج المدرب في model/cnn_model.h5")
