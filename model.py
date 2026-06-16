import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from typing import Tuple, Optional, Dict, Any


def build_cnn(
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    dropout_rate: float = 0.5,
    l2_reg: float = 0.0001,
    conv_filters: Tuple[int, ...] = (32, 64, 128),
    kernel_size: Tuple[int, int] = (3, 3),
    pool_size: Tuple[int, int] = (2, 2),
    dense_units: int = 256,
) -> tf.keras.Model:
    model = models.Sequential(name='digit_cnn')
    model.add(layers.Input(shape=input_shape))

    for i, filters in enumerate(conv_filters):
        model.add(layers.Conv2D(
            filters, kernel_size, padding='same',
            kernel_regularizer=regularizers.l2(l2_reg),
            activation='relu'
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(
            filters, kernel_size, padding='same',
            kernel_regularizer=regularizers.l2(l2_reg),
            activation='relu'
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=pool_size))
        model.add(layers.Dropout(dropout_rate * 0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 0.001,
    optimizer: str = 'adam',
) -> tf.keras.Model:
    opt = {
        'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
        'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
    }.get(optimizer, tf.keras.optimizers.Adam(learning_rate=learning_rate))

    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def get_callbacks(
    patience: int = 5,
    model_path: str = 'models/best_model.h5',
) -> list:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=patience,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
    ]


def create_data_augmentation():
    return tf.keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomBrightness(0.05),
    ])


class DigitRecognitionModel:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model: Optional[tf.keras.Model] = None
        self.history: Optional[tf.keras.callbacks.History] = None

    def build(self) -> 'DigitRecognitionModel':
        self.model = build_cnn(**self.config)
        compile_model(self.model, **self.config)
        return self

    def train(
        self,
        x_train, y_train,
        x_val, y_val,
        epochs: int = 50,
        batch_size: int = 128,
        use_augmentation: bool = True,
    ) -> 'DigitRecognitionModel':
        callbacks = get_callbacks(
            patience=self.config.get('patience', 5),
            model_path=self.config.get('model_path', 'models/best_model.h5'),
        )

        if use_augmentation:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10,
                zoom_range=0.1,
                width_shift_range=0.1,
                height_shift_range=0.1,
                brightness_range=(0.8, 1.2),
            )
            datagen.fit(x_train)
            self.history = self.model.fit(
                datagen.flow(x_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(x_train) // batch_size,
                validation_data=(x_val, y_val),
                epochs=epochs,
                callbacks=callbacks,
                verbose=1,
            )
        else:
            self.history = self.model.fit(
                x_train, y_train,
                batch_size=batch_size,
                validation_data=(x_val, y_val),
                epochs=epochs,
                callbacks=callbacks,
                verbose=1,
            )
        return self

    def predict(self, x) -> Tuple[np.ndarray, np.ndarray]:
        import numpy as np
        probs = self.model.predict(x, verbose=0)
        classes = np.argmax(probs, axis=1)
        return classes, probs

    def evaluate(self, x_test, y_test) -> Dict[str, float]:
        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        return {'loss': loss, 'accuracy': acc}

    def save(self, filepath: str) -> 'DigitRecognitionModel':
        self.model.save(filepath)
        return self

    def load(self, filepath: str) -> 'DigitRecognitionModel':
        self.model = tf.keras.models.load_model(filepath)
        return self

    def summary(self):
        return self.model.summary()
