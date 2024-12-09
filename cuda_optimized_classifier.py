import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout, 
    BatchNormalization, LeakyReLU, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import os

# Enable memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"Found {len(physical_devices)} GPU(s). Memory growth enabled.")
else:
    print("No GPU found. Running on CPU.")

# Set TensorFlow to use mixed precision for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Constants
IMAGE_SIZE = 128
BATCH_SIZE = 32  # Increased batch size for GPU
LEARNING_RATE = 0.001
AUTOTUNE = tf.data.AUTOTUNE

def build_cuda_optimized_cnn(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=7):
    """
    Build a CUDA-optimized CNN architecture
    """
    model = Sequential([
        # First Block - Initial Feature Extraction
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Second Block
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Third Block
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        GlobalAveragePooling2D(),
        
        Dense(256),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),

        # Output layer with float32 for stability
        Dense(num_classes, activation='sigmoid', dtype='float32')
    ])

    return model

# Optimized data processing pipeline using tf.data
def create_dataset(df, x_col, y_col, batch_size, is_training=True):
    """
    Create an optimized tf.data.Dataset pipeline
    """
    def process_path(path, labels):
        # Read and decode image
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        
        return img, labels

    def augment(image, label):
        # Data augmentation using tf.image for GPU acceleration
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_flip_left_right(image)
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_brightness(image, 0.2)
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_contrast(image, 0.8, 1.2)
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_saturation(image, 0.8, 1.2)
            
        return image, label

    # Create tf.data.Dataset
    paths = df[x_col].values
    labels = df[y_col].values
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    
    if is_training:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset

# Custom F1 Score Metric optimized for GPU
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        self.true_positives.assign_add(tf.reduce_sum(y_true * y_pred))
        self.false_positives.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.false_negatives.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        return 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

    def reset_state(self):
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)

def train_model_with_cuda(model, train_dataset, val_dataset, epochs=50):
    """
    Train the model with CUDA optimizations
    """
    # Use gradient accumulation for larger effective batch size
    gradient_accumulation_steps = 2
    
    callbacks = [
        EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_f1_score',
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model_cuda.h5',
            monitor='val_f1_score',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]

    # Compile with mixed precision optimizer
    optimizer = Adam(LEARNING_RATE)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', F1Score()]
    )

    # Train with GPU optimization
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=4
    )

    return history

# Example usage:
"""
# Prepare your data
label_columns = ['Hoodie', 'Kaos', 'biru', 'hitam', 'kuning', 'merah', 'putih']

# Create optimized datasets
train_dataset = create_dataset(
    traindf,
    x_col='path',
    y_col=label_columns,
    batch_size=BATCH_SIZE,
    is_training=True
)

val_dataset = create_dataset(
    valdf,
    x_col='path',
    y_col=label_columns,
    batch_size=BATCH_SIZE,
    is_training=False
)

# Build and train model
model = build_cuda_optimized_cnn(num_classes=len(label_columns))
history = train_model_with_cuda(model, train_dataset, val_dataset)
"""
