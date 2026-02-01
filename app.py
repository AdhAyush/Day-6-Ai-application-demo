from flask import Flask, render_template, request, jsonify
from model import predict_digit
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle canvas drawing
    if 'image' in request.json:
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])
        image = Image.open(BytesIO(image_data))
    
    digit, confidence = predict_digit(image)
    return jsonify({'digit': int(digit), 'confidence': float(confidence)})

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    # Handle file upload
    file = request.files['file']
    image = Image.open(file.stream)
    
    digit, confidence = predict_digit(image)
    return jsonify({'digit': int(digit), 'confidence': float(confidence)})

if __name__ == '__main__':
    app.run(debug=True)