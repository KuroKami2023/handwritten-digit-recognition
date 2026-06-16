"""
Evaluate the trained digit recognition model on the test set.
Produces confusion matrix, classification report, and misclassification examples.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from model import DigitRecognitionModel

MODELS_DIR = 'models'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'digit_cnn_final.h5')


def load_test_data():
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1)
    return x_test, y_test


def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=range(10), yticklabels=range(10))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix - MNIST CNN')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Confusion matrix saved to {save_path}')


def plot_misclassifications(x_test, y_true, y_pred, num_samples=25, save_path='results/misclassifications.png'):
    misclassified = np.where(y_true != y_pred)[0]
    if len(misclassified) == 0:
        print('No misclassifications found!')
        return
    selected = np.random.choice(misclassified, min(num_samples, len(misclassified)), replace=False)
    cols = 5
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    for i, idx in enumerate(selected):
        ax = axes[i]
        ax.imshow(x_test[idx].squeeze(), cmap='gray')
        ax.set_title(f'True: {y_true[idx]}\nPred: {y_pred[idx]}', color='red', fontsize=10)
        ax.axis('off')
    for i in range(len(selected), len(axes)):
        axes[i].axis('off')
    plt.suptitle(f'Misclassified Samples ({len(misclassified)} total)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Misclassification examples saved to {save_path}')


def plot_class_accuracies(y_true, y_pred, save_path='results/class_accuracies.png'):
    cm = confusion_matrix(y_true, y_pred)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(10), class_acc, color='steelblue')
    ax.set_xlabel('Digit')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy')
    ax.set_xticks(range(10))
    ax.set_ylim(0, 1.05)
    for bar, acc in zip(bars, class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Class accuracies saved to {save_path}')
    return class_acc


def main():
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f'Error: No trained model found at {DEFAULT_MODEL_PATH}')
        print('Run train_cnn.py first!')
        sys.exit(1)
    print('=' * 60)
    print('Model Evaluation')
    print('=' * 60)
    print(f'\nLoading model from {DEFAULT_MODEL_PATH}...')
    model = DigitRecognitionModel()
    model.load(DEFAULT_MODEL_PATH)
    print('Loading test data...')
    x_test, y_test = load_test_data()
    print('Making predictions...')
    y_pred, y_probs = model.predict(x_test)
    loss, acc = model.model.evaluate(x_test, y_test, verbose=0)
    print(f'\nTest Loss: {loss:.4f}')
    print(f'Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, digits=4))
    plot_confusion_matrix(y_test, y_pred)
    plot_misclassifications(x_test, y_test, y_pred)
    plot_class_accuracies(y_test, y_pred)
    misclassified_count = np.sum(y_test != y_pred)
    print(f'\nTotal misclassified: {misclassified_count} / {len(y_test)} ({misclassified_count / len(y_test) * 100:.2f}%)')
    print(f'\nAll evaluation artifacts saved to {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
