"""
Simple Flask web app for handwritten digit recognition.
Users draw on a canvas and the trained CNN predicts the digit.
"""

import os
import sys
import base64
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template

from model import DigitRecognitionModel

app = Flask(__name__)
MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'digit_cnn_final.h5')

model = None


def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f'Model not found at {MODEL_PATH}. Training in-memory fallback...')
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_train = np.expand_dims(x_train, axis=-1)
        model = DigitRecognitionModel({
            'input_shape': (28, 28, 1),
            'num_classes': 10,
            'dropout_rate': 0.3,
            'conv_filters': (32, 64),
            'dense_units': 128,
        })
        model.build()
        model.model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=1)
        os.makedirs(MODELS_DIR, exist_ok=True)
        model.save(MODEL_PATH)
        print(f'Model saved to {MODEL_PATH}')
    else:
        model = DigitRecognitionModel()
        model.load(MODEL_PATH)
        print(f'Model loaded from {MODEL_PATH}')


def preprocess_canvas_image(image_data: str) -> np.ndarray:
    header, encoded = image_data.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = 1.0 - img_array
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        x = preprocess_canvas_image(data['image'])
        classes, probs = model.predict(x)
        predicted_class = int(classes[0])
        confidence = float(probs[0][predicted_class])
        probabilities = [float(p) for p in probs[0]]
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200


def main():
    load_model()
    print('Starting web app at http://127.0.0.1:5000')
    print('Draw a digit on the canvas and click Predict!')
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()
