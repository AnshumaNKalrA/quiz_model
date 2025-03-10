import requests
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
    student_answers = pdf_processing(pages)
    # Transform data to desired format
    print(student_answers)
    transformed_data = []
    for student_id, answers in student_answers.items():
        # Convert numeric answers to strings
        string_answers = [str(answer) for answer in answers]
        
        student_record = {
            "playerId": int(student_id),
            "quizId": 68,
            "markedResponses": string_answers,
            "score": 0  # Set default score to 0
        }
        transformed_data.append(student_record)
    
    # Return the transformed data as JSON
    print(transformed_data)
    # Send POST requests one by one
    url = "http://localhost:8080/playerResponse"
    responses = []
    for data in transformed_data:
        try:
            response = requests.post(url, json=data)
            responses.append({
                "playerId": data["playerId"],
                "status": response.status_code,
                "response": response.text
            })
        except requests.exceptions.RequestException as e:
            responses.append({
                "playerId": data["playerId"],
                "status": "Failed",
                "response": str(e)
            })
    print(responses)
    return jsonify(responses)

if __name__ == "__main__":
    app.run(debug=True)