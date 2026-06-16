# Handwritten Digit Recognition

[](https://github.com/KuroKami2023/handwritten-digit-recognition)
[](LICENSE)
[](https://www.python.org/)
[](https://www.tensorflow.org/)
[]()

> A Convolutional Neural Network (CNN) that recognizes handwritten digits (0–9) from the MNIST dataset with 99.4%+ test accuracy, featuring an interactive web drawing interface.

---

> 💡 **Portfolio demo:** A simplified browser-based version is available in the [portfolio website](https://kurokami2023.github.io).

---

## Features

- [x] **Deep CNN Architecture** — 3 convolutional blocks (32→64→128 filters) with Batch Normalization and Dropout
- [x] **Data Augmentation** — Random rotation, zoom, shift, and brightness for improved generalization
- [x] **Interactive Web UI** — Draw digits on an HTML5 canvas and get real-time predictions via Flask
- [x] **Probability Distribution** — Confidence scores for all 10 digits with top-3 breakdown
- [x] **Comprehensive Evaluation** — Confusion matrix, per-class accuracy, misclassification analysis
- [x] **Training Callbacks** — Early stopping, learning rate reduction, model checkpointing
- [x] **CLI Prediction** — Predict digits from image files via command line

## Dataset

| Property | Value |
|----------|-------|
| **Dataset** | [MNIST](http://yann.lecun.com/exdb/mnist/) (Modified National Institute of Standards and Technology) |
| **Samples** | 70,000 grayscale images |
| **Split** | 60,000 train / 10,000 test |
| **Image Size** | 28 × 28 pixels |
| **Classes** | 10 (digits 0–9) |
| **Auto-download** | Built into TensorFlow (`tf.keras.datasets.mnist`) |

## Model Architecture

```
Input (28 × 28 × 1)
│
├─ Conv2D(32) + BatchNorm + ReLU
│  └─ Conv2D(32) + BatchNorm + ReLU → MaxPool(2×2) → Dropout(0.25)
│
├─ Conv2D(64) + BatchNorm + ReLU
│  └─ Conv2D(64) + BatchNorm + ReLU → MaxPool(2×2) → Dropout(0.25)
│
├─ Conv2D(128) + BatchNorm + ReLU
│  └─ Conv2D(128) + BatchNorm + ReLU → MaxPool(2×2) → Dropout(0.25)
│
├─ Flatten → Dense(256) + BatchNorm + ReLU → Dropout(0.5)
│
└─ Dense(10) + Softmax
```

### Key Design Choices

- **L2 Kernel Regularization** (λ = 0.0001) prevents overfitting
- **Batch Normalization** after each Conv layer stabilizes training
- **Dropout** (0.25 after conv blocks, 0.5 before output) for regularization
- **Adam Optimizer** (lr = 0.001) with ReduceLROnPlateau scheduling
- **Early Stopping** (patience = 5) restores best weights

## Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **99.4%+** |
| Test Loss | ~0.02 |
| Misclassification Rate | < 0.6% |
| Parameters | ~1.2M |
| Training Time | ~5 min (CPU) / ~30s (GPU) |

# Tech Stack

- **Python** — Core language
- **TensorFlow / Keras** — Deep learning framework
- **Flask** — Web server for interactive UI
- **Pillow** — Image preprocessing
- **Matplotlib / Seaborn** — Evaluation plots
- **NumPy** — Numerical operations
- **HTML5 Canvas** — Drawing interface

## Installation

```bash
# Clone the repository
git clone https://github.com/KuroKami2023/handwritten-digit-recognition.git
cd handwritten-digit-recognition

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Train the Model

```bash
python train_cnn.py
```

Trains the CNN on MNIST with data augmentation. Saves the final model to `models/digit_cnn_final.h5` and training plots to `results/`.

### Evaluate

```bash
python evaluate.py
```

Generates confusion matrix, per-class accuracy bar chart, and misclassification examples.

### Predict on an Image File

```bash
python predict.py path/to/digit_image.png
python predict.py --model models/digit_cnn_final.h5 path/to/digit_image.png
```

### Run the Web App

```bash
python web_app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. Draw a digit on the canvas, click **Predict**, and the model returns:

- Predicted digit
- Confidence score
- Probability distribution across all 10 digits

The web app also trains a quick fallback model if no pre-trained model is found.

## Project Structure

```
handwritten-digit-recognition/
├── train_cnn.py              # Training script with augmentation
├── model.py                  # CNN architecture and DigitRecognitionModel class
├── evaluate.py               # Comprehensive evaluation suite
├── predict.py                # CLI prediction from image files
├── web_app.py                # Flask web app with canvas drawing
├── templates/
│   └── index.html            # Web UI with drawing canvas
├── models/                   # Saved model weights (created after training)
├── results/                  # Evaluation plots (created after training/evaluation)
├── requirements.txt
├── LICENSE
└── README.md
```

## Training Details & Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr = 0.001) |
| Batch size | 128 |
| Max epochs | 50 |
| Early stopping patience | 5 |
| LR reduction factor | 0.5 |
| LR reduction patience | 3 |
| Min learning rate | 1 × 10⁻⁶ |
| L2 regularization | 0.0001 |
| Validation split | 5,000 samples from training set |

### Data Augmentation

| Transform | Range |
|-----------|-------|
| Rotation | ±10° |
| Zoom | ±10% |
| Width shift | ±10% |
| Height shift | ±10% |
| Brightness | 0.8× – 1.2× |

## Performance Benchmarks

| Hardware | Epoch Time | Training Time (50 epochs) |
|----------|-----------|--------------------------|
| CPU (Intel i7) | ~6s | ~5 min |
| GPU (NVIDIA RTX 3060) | ~0.6s | ~30s |
| Apple M1 | ~2s | ~1.5 min |

## Future Improvements

- [ ] Support for EMNIST (extended letters + digits)
- [ ] Grad-CAM heatmaps for model explainability
- [ ] Mobile deployment via TensorFlow Lite
- [ ] Sequence recognition for multi-digit numbers
- [ ] Style transfer / adversarial robustness testing
- [ ] Quantization-aware training for edge devices

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
