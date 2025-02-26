import cv2
import numpy as np
from encryption import encrypt_message, decrypt_message

def embed_message(image_path, message, output_image):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    max_bytes = (h * w * 3) // 8 

    encrypted_msg = encrypt_message(message)  # تشفير الرسالة
    binary_msg = ''.join(format(byte, '08b') for byte in encrypted_msg)

    if len(binary_msg) > max_bytes:
        raise ValueError(f"The message is too large! Max allowed: {max_bytes} bytes, got {len(binary_msg)//8} bytes.")

    # تضمين طول الرسالة في أول 32 بت
    msg_length_bin = format(len(binary_msg), '032b')
    binary_msg = msg_length_bin + binary_msg 

    data_index = 0
    for i in range(h):
        for j in range(w):
            for k in range(3):
                if data_index < len(binary_msg):
                    img[i, j, k] = (img[i, j, k] & 0xFE) | int(binary_msg[data_index])
                    data_index += 1
                else:
                    break

    cv2.imwrite(output_image, img)
    print("✅ Message successfully embedded!")

def extract_message(image_path, private_key_path="private.pem"):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    binary_msg = ""

    data_index = 0
    for i in range(h):
        for j in range(w):
            for k in range(3):
                binary_msg += str(img[i, j, k] & 1)
                data_index += 1
                if data_index >= 32 and data_index >= int(binary_msg[:32], 2) + 32:
                    break

    msg_length = int(binary_msg[:32], 2)
    binary_msg = binary_msg[32:32 + msg_length]

    byte_data = [int(binary_msg[i:i+8], 2) for i in range(0, len(binary_msg), 8)]
    encrypted_msg = bytes(byte_data)

    if len(encrypted_msg) < 128:  # الحد الأدنى لطول الرسالة
        return "Error: Extracted message is too short."

    try:
        decrypted_msg = decrypt_message(encrypted_msg, private_key_path)
        return decrypted_msg
    except Exception as e:
        return f"Decryption failed: {str(e)}"
