"""
Train a CNN on the MNIST dataset for handwritten digit recognition.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from model import DigitRecognitionModel

MODELS_DIR = 'models'
RESULTS_DIR = 'results'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_mnist_data():
    print('Loading MNIST dataset...')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    val_size = 5000
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train = x_train[val_size:]
    y_train = y_train[val_size:]
    print(f'Training samples: {len(x_train)}')
    print(f'Validation samples: {len(x_val)}')
    print(f'Test samples: {len(x_test)}')
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def plot_training_history(history, save_path='results/training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Training history saved to {save_path}')


def plot_predictions(model, x_test, y_test, num_samples=20, save_path='results/sample_predictions.png'):
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    x_sample = x_test[indices]
    y_sample = y_test[indices]
    preds, probs = model.predict(x_sample)
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_sample[i].squeeze(), cmap='gray')
        color = 'green' if preds[i] == y_sample[i] else 'red'
        ax.set_title(f'True: {y_sample[i]}\nPred: {preds[i]} ({probs[i][preds[i]]:.2f})',
                     color=color, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Sample predictions saved to {save_path}')


def main():
    print('=' * 60)
    print('MNIST Digit Recognition - CNN Training')
    print('=' * 60)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_data()
    print('\nBuilding CNN model...')
    config = {
        'input_shape': (28, 28, 1),
        'num_classes': 10,
        'dropout_rate': 0.5,
        'l2_reg': 0.0001,
        'conv_filters': (32, 64, 128),
        'dense_units': 256,
        'learning_rate': 0.001,
        'optimizer': 'adam',
    }
    model = DigitRecognitionModel(config)
    model.build()
    print('\nModel summary:')
    model.summary()
    print('\nTraining model...')
    model.train(
        x_train, y_train,
        x_val, y_val,
        epochs=50,
        batch_size=128,
        use_augmentation=True,
    )
    model_path = os.path.join(MODELS_DIR, 'digit_cnn_final.h5')
    model.save(model_path)
    print(f'\nModel saved to {model_path}')
    plot_training_history(model.history)
    eval_results = model.evaluate(x_test, y_test)
    print(f'\nTest Results: Loss={eval_results["loss"]:.4f}, Accuracy={eval_results["accuracy"]:.4f}')
    plot_predictions(model, x_test, y_test)
    return 0


if __name__ == '__main__':
    sys.exit(main())
