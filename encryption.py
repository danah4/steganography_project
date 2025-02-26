import rsa
import os

KEY_SIZE = 2048

def generate_rsa_keys():
    if not os.path.exists("public.pem") or not os.path.exists("private.pem"):
        public_key, private_key = rsa.newkeys(KEY_SIZE)
        with open("public.pem", "wb") as pub_file:
            pub_file.write(public_key.save_pkcs1("PEM"))
        with open("private.pem", "wb") as priv_file:
            priv_file.write(private_key.save_pkcs1("PEM"))
        print("✅ RSA keys generated successfully!")

def encrypt_message(message, public_key_path="public.pem"):
    with open(public_key_path, "rb") as pub_file:
        public_key = rsa.PublicKey.load_pkcs1(pub_file.read())

    max_msg_length = (KEY_SIZE // 8) - 11  # الحد الأقصى للبيانات القابلة للتشفير
    if len(message.encode()) > max_msg_length:
        raise ValueError(f"Message too long! Max allowed: {max_msg_length} bytes, but got {len(message.encode())} bytes.")

    encrypted_msg = rsa.encrypt(message.encode(), public_key)
    return encrypted_msg

def decrypt_message(encrypted_msg, private_key_path="private.pem"):
    with open(private_key_path, "rb") as priv_file:
        private_key = rsa.PrivateKey.load_pkcs1(priv_file.read())

    try:
        decrypted_msg = rsa.decrypt(encrypted_msg, private_key).decode()
        return decrypted_msg
    except rsa.DecryptionError:
        return "Decryption failed: The message may be corrupted or the keys do not match."

# إنشاء المفاتيح مرة واحدة فقط
generate_rsa_keys()
