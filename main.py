from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import tempfile

app = Flask(__name__, static_folder='.', static_url_path='', template_folder='.')

# Load the trained model
model_path = 'skin_cancer_cnn.h5'
if not os.path.exists(model_path):
    print(f"❌ Model file '{model_path}' not found. Please train the model first.")
    exit(1)

model = load_model(model_path)
print(f"✅ Model loaded from: {model_path}")

# Function to preprocess and predict the image
def predict_skin_cancer(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))  # Load Image
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make Prediction
    prediction = model.predict(img_array, verbose=0)
    confidence = float(prediction[0, 0])
    class_label = "Malignant" if confidence > 0.5 else "Benign"
    confidence_score = confidence if confidence > 0.5 else 1 - confidence

    return class_label, confidence_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save the file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name
    
    try:
        # Predict
        class_label, confidence_score = predict_skin_cancer(temp_path, model)
        
        # Clean up
        os.unlink(temp_path)
        
        return jsonify({
            'prediction': class_label,
            'confidence': f"{confidence_score:.2%}"
        })
    except Exception as e:
        os.unlink(temp_path)
        return jsonify({'error': str(e)}), 500

import threading
import webbrowser

if __name__ == "__main__":
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        threading.Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    app.run(debug=True)
