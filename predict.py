"""
Load a trained model and predict on new digit images.
Supports drawing via command line or loading an image file.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import DigitRecognitionModel

MODELS_DIR = 'models'
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'digit_cnn_final.h5')


def preprocess_image(image_path: str, size: tuple = (28, 28)) -> np.ndarray:
    img = Image.open(image_path).convert('L')
    img = img.resize(size, Image.Resampling.LANCZOS)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_from_file(image_path: str, model_path: str = DEFAULT_MODEL_PATH):
    if not os.path.exists(model_path):
        print(f'Error: Model not found at {model_path}')
        print('Run train_cnn.py first to train and save a model.')
        return
    print(f'Loading model from {model_path}...')
    model = DigitRecognitionModel()
    model.load(model_path)
    print('Preprocessing image...')
    x = preprocess_image(image_path)
    classes, probs = model.predict(x)
    predicted_class = classes[0]
    confidence = probs[0][predicted_class]
    print(f'\nPrediction: {predicted_class} (confidence: {confidence:.4f})')
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    original = Image.open(image_path).convert('L')
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    processed = x[0].squeeze()
    axes[1].imshow(processed, cmap='gray')
    axes[1].set_title(f'Processed ({predicted_class})')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('results/prediction_result.png', dpi=150)
    plt.show()
    print('\nTop-3 predictions:')
    top3 = np.argsort(probs[0])[::-1][:3]
    for i, cls in enumerate(top3):
        print(f'  {i+1}. Digit {cls}: {probs[0][cls]:.4f}')


def main():
    if len(sys.argv) < 2:
        print('Usage:')
        print('  python predict.py <image_path>')
        print('  python predict.py --model <model_path> <image_path>')
        sys.exit(1)
    model_path = DEFAULT_MODEL_PATH
    image_path = sys.argv[-1]
    if '--model' in sys.argv:
        idx = sys.argv.index('--model')
        model_path = sys.argv[idx + 1]
        image_path = sys.argv[-1]
    if not os.path.exists(image_path):
        print(f'Error: Image not found at {image_path}')
        sys.exit(1)
    predict_from_file(image_path, model_path)


if __name__ == '__main__':
    main()
