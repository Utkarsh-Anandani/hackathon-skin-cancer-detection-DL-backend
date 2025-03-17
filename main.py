from tensorflow.keras.models import load_model
import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify

model = load_model('my_skin_disease_pred_model.h5')

# Define class names
class_names = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        print(data)
        file_url = data.get('image')

        if not file_url:
            return jsonify({'error': 'No file URL provided'}), 400

        try:
            predicted_class_index, predicted_class_name, predicted_prob = predict_image_from_url(file_url)
            res = {
                'predicted_class': predicted_class_name,
                'prediction_probability': str(predicted_prob)
            }
            return jsonify(res)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

def predict_image_from_url(file_url):
    response = requests.get(file_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch image from URL: {file_url}")

    file_bytes = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    image = cv2.resize(image, (64, 64))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_prob = predictions[0][predicted_class_index]
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_index, predicted_class_name, predicted_prob

if __name__ == '__main__':
    app.run(port=5000)
