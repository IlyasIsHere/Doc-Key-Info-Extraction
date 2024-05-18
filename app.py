from flask import Flask, request, jsonify, send_file, render_template
import os
from PIL import Image
import model
import pytesseract
import llm 

app = Flask(__name__)

from transformers import pipeline

pipe = pipeline("document-question-answering", model="impira/layoutlm-document-qa")
questions=open("fields.txt", "r")
questions=[question.strip() for question in questions]
print(questions)

def save_image(image, filename):
    input_folder = "input"
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    image_path = os.path.join(input_folder, filename)
    image.save(image_path)
    return image_path


def export_data(data, filename="processed_data.txt"):
    # Export processed data to a text file for download
    with open(filename, "w") as file:
        for key, value in data.items():
            file.write(f"{key}: {value}\n")
    return filename


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



def ocr():
    uploaded_file = request.files['file']
    model_type = request.form.get('model', 'gemini')
    if uploaded_file.filename != '':
        image = Image.open(uploaded_file)
        image_path = save_image(image, uploaded_file.filename)
        processed_data = model.process(image, model=model_type)
        return jsonify(processed_data)
    else:
        return "No file uploaded", 400



@app.route('/process', methods=['POST'])
def process():
    uploaded_file = request.files.get('file')
    prompt = request.form.get('prompt', None)
    model_type = request.form.get('model', 'gemini')
    print("Model type: ", model_type)
    if uploaded_file and uploaded_file.filename != '':
        if model_type == 'gemini':
            image = Image.open(uploaded_file)
            image_path = save_image(image, uploaded_file.filename)
            processed_data = llm.process(image, prompt=prompt)
            return jsonify({"result": processed_data})
        elif model_type == 'layoutlmv1':
            image=Image.open(uploaded_file)
            image_path=save_image(image, uploaded_file.filename)
            print(image_path)
            print(prompt)
            if prompt is None or prompt=="":
                answer={}
                for question in questions:
                    print(question)
                    result=pipe(image=image_path, 
                                question=question)
                    result=result[0]['answer']
                    answer[question]=result
                    print(answer)
                return jsonify({"result": str(answer)})
            else:
                result=pipe(image=image_path, 
                            question=prompt)
                result=result[0]['answer']
                return jsonify({"result": result})
        elif model_type == 'layoutlmv2':
            return jsonify({"result": "LayoutLMv2 results"})
        elif model_type == 'ocr':
            text = pytesseract.image_to_string(uploaded_file)
            return jsonify({"result": text})
        else:
            print("Invalid model type")
            return jsonify({"error": "Invalid model type"})
    else:
        print("No file uploaded")
        return jsonify({"error": "No file uploaded"}), 400



@app.route('/prompt', methods=['POST'])
def prompt():
    uploaded_file = request.files['file']
    model_type = request.form.get('model', 'gemini')
    if uploaded_file.filename != '':
        image = Image.open(uploaded_file)
        image_path = save_image(image, uploaded_file.filename)
        return "Prompt for additional input"
    else:
        return "No file uploaded", 400



@app.route('/information', methods=['POST'])
def information():
    data = request.json
    if data:
        model_type = data.get('model', 'gemini')
        processed_data = model.process(data['image'], prompt=data.get('prompt', None), model=model_type)
        filename = export_data(processed_data)
        return send_file(filename, as_attachment=True)
    else:
        return "No data provided", 400




@app.route('/pipeline', methods=['POST'])
def pipeline():
    uploaded_file = request.files['file']
    prompt = request.form.get('prompt', None)
    model_type = request.form.get('model', 'gemini')
    if uploaded_file.filename != '':
        image = Image.open(uploaded_file)
        image_path = save_image(image, uploaded_file.filename)
        if prompt:
            return "Prompt for additional input"
        else:
            processed_data = model.process(image, model=model_type)
            return jsonify(processed_data)
    else:
        return "No file uploaded", 400




if __name__ == "__main__":
    app.run(debug=True)