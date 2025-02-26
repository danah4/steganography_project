from flask import Flask, request, jsonify, send_file
from steg import embed_message, extract_message  # تأكد من أن لديك steg.py

app = Flask(__name__)

@app.route('/embed', methods=['POST'])
def embed():
    file = request.files['image']
    message = request.form['message']
    output_path = "stego_image.png"
    file.save("input.png")  # احفظ الصورة المرفوعة
    embed_message("input.png", message, output_path)  # نفذ التضمين
    return send_file(output_path, as_attachment=True)  # أرسل الصورة الجديدة

@app.route('/extract', methods=['POST'])
def extract():
    file = request.files['image']
    file.save("stego_input.png")  # احفظ الصورة المرفوعة
    extracted_msg = extract_message("stego_input.png")  # نفذ استخراج الرسالة
    return jsonify({"Extracted Message": extracted_msg})

if __name__ == '__main__':
    app.run(debug=True)