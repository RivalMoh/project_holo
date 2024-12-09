import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import albumentations as A

# Improved image size for better feature extraction
IMAGE_SIZE = 224  # EfficientNet recommended size

# Advanced augmentation pipeline
def get_augmentation():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ])

# Custom data generator with Albumentations
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, x_col, y_col, batch_size, target_size, shuffle=True, augment=False):
        self.df = df
        self.x_col = x_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augment = augment
        self.aug = get_augmentation() if augment else None
        self.n = len(self.df)
        self.indexes = np.arange(len(df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

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
            
            if self.augment:
                augmented = self.aug(image=img)
                img = augmented['image']
            
            batch_x[i] = img / 255.0
            batch_y[i] = self.df.iloc[idx][self.y_col].values
        
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Improved model architecture using EfficientNet
def build_model(num_classes):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    
    # Fine-tune the last few layers
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

# Custom F1 score metric
def f1_score(y_true, y_pred):
    y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return f1

# Training function with learning rate warmup and cosine decay
def train_model(model, train_gen, val_gen, epochs=50):
    initial_learning_rate = 1e-4
    warmup_epochs = 5
    
    def cosine_decay_with_warmup(epoch):
        if epoch < warmup_epochs:
            return initial_learning_rate * (epoch + 1) / warmup_epochs
        else:
            return initial_learning_rate * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    callbacks = [
        EarlyStopping(monitor='val_f1_score', mode='max', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_f1_score', mode='max', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint('best_model.h5', monitor='val_f1_score', mode='max', save_best_only=True),
        tf.keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup)
    ]
    
    model.compile(
        optimizer=Adam(initial_learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', f1_score]
    )
    
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
# Load and preprocess your data
df = pd.read_csv('your_data.csv')
mlb = MultiLabelBinarizer()
label_columns = ['Hoodie', 'Kaos', 'biru', 'hitam', 'kuning', 'merah', 'putih']

# Split the data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Create data generators
train_gen = CustomDataGenerator(
    train_df,
    x_col='path',
    y_col=label_columns,
    batch_size=16,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    augment=True
)

val_gen = CustomDataGenerator(
    val_df,
    x_col='path',
    y_col=label_columns,
    batch_size=16,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    augment=False
)

# Build and train model
model = build_model(num_classes=len(label_columns))
history = train_model(model, train_gen, val_gen)
"""
