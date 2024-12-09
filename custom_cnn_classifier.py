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

# Constants
IMAGE_SIZE = 128
BATCH_SIZE = 16
LEARNING_RATE = 0.001

def build_custom_cnn(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=7):
    """
    Build a custom CNN architecture with:
    - Deeper network with residual-like connections
    - Batch normalization for better training stability
    - LeakyReLU for better gradient flow
    - Dropout for regularization
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

        # Second Block - Increased Channels
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Third Block - Deep Feature Extraction
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Fourth Block - Final Feature Extraction
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Global Average Pooling instead of Flatten
        GlobalAveragePooling2D(),

        # Dense layers for classification
        Dense(512),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        
        Dense(256),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),

        # Output layer
        Dense(num_classes, activation='sigmoid')
    ])

    return model

# Custom F1 Score Metric
def f1_score(y_true, y_pred):
    """Calculate F1 score for multi-label classification"""
    y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
    
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)

# Advanced Data Generator
class AdvancedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, x_col, y_col, batch_size, target_size, is_training=True):
        self.df = df
        self.x_col = x_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.target_size = target_size
        self.is_training = is_training
        self.n = len(self.df)
        self.indexes = np.arange(len(df))
        
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.n)
        batch_indexes = self.indexes[start_idx:end_idx]
        
        batch_x = np.zeros((len(batch_indexes), *self.target_size, 3))
        batch_y = np.zeros((len(batch_indexes), len(self.y_col)))
        
        for i, idx in enumerate(batch_indexes):
            img = cv2.imread(self.df.iloc[idx][self.x_col])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.target_size[:2])
            
            if self.is_training:
                # Apply augmentations
                if np.random.random() > 0.5:
                    img = cv2.flip(img, 1)  # horizontal flip
                
                if np.random.random() > 0.5:
                    angle = np.random.uniform(-15, 15)
                    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1)
                    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                
                # Random brightness and contrast
                if np.random.random() > 0.5:
                    alpha = np.random.uniform(0.8, 1.2)  # contrast
                    beta = np.random.uniform(-10, 10)    # brightness
                    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
            batch_x[i] = img / 255.0
            batch_y[i] = self.df.iloc[idx][self.y_col].values
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)

def train_custom_model(model, train_gen, val_gen, epochs=50):
    """
    Train the model with advanced training techniques
    """
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when training plateaus
        ReduceLROnPlateau(
            monitor='val_f1_score',
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # Save best model
        ModelCheckpoint(
            'best_custom_model.h5',
            monitor='val_f1_score',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]

    # Compile model with custom metric
    model.compile(
        optimizer=Adam(LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', f1_score]
    )

    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True
    )

    return history

# Example usage:
"""
# Prepare your data
label_columns = ['Hoodie', 'Kaos', 'biru', 'hitam', 'kuning', 'merah', 'putih']

# Create data generators
train_gen = AdvancedDataGenerator(
    traindf,
    x_col='path',
    y_col=label_columns,
    batch_size=BATCH_SIZE,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    is_training=True
)

val_gen = AdvancedDataGenerator(
    valdf,
    x_col='path',
    y_col=label_columns,
    batch_size=BATCH_SIZE,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    is_training=False
)

# Build and train model
model = build_custom_cnn(num_classes=len(label_columns))
history = train_custom_model(model, train_gen, val_gen)
"""
