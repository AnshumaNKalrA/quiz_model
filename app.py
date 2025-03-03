import os
import uuid
from flask import Flask,request,jsonify
from pdf2image import convert_from_bytes
from main import pdf_processing

app = Flask(__name__)

@app.route('/process_pdf',methods=['POST'])
def process_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No PDF file uploaded."}), 400

    pdf_file = request.files['pdf_file']
    pdf_bytes = pdf_file.read()

    if not pdf_bytes:
        return jsonify({'error':'Empty PDF file'}), 400

    pages = convert_from_bytes(pdf_bytes, dpi=144)
    json_to_return = pdf_processing(pages)

    return json_to_return

if __name__ == "__main__":
    app.run(debug=True)