# Multi-Label Clothing Classification Project

This project implements a multi-label image classification system for clothing items, capable of identifying both the type of clothing (Hoodie/Kaos) and its color (blue, black, yellow, red, white). The project includes three different implementations, each with its own advantages and optimizations.

## Project Structure

```
holo/
│
├── improved_clothing_classifier.py    # Implementation using EfficientNet
├── custom_cnn_classifier.py          # Custom CNN implementation from scratch
├── cuda_optimized_classifier.py      # CUDA-optimized version for GPU acceleration
└── README.md                         # This documentation file
```

## 1. Improved Clothing Classifier (`improved_clothing_classifier.py`)

This implementation uses a pre-trained EfficientNetB0 model as the base architecture.

### Key Components:

#### Data Augmentation
```python
def get_augmentation():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        # ... more augmentations
    ])
```
- Uses Albumentations library for sophisticated image augmentation
- Includes multiple augmentation techniques for better generalization
- Probability-based application of transformations

#### Custom Data Generator
```python
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, x_col, y_col, batch_size, target_size, shuffle=True, augment=False):
        # ... initialization
```
- Inherits from `tf.keras.utils.Sequence`
- Handles batch generation and data augmentation
- Memory-efficient implementation

#### Model Architecture
- Base: EfficientNetB0 pre-trained on ImageNet
- Additional layers:
  - Global Average Pooling
  - Batch Normalization
  - Dense layers with dropout
  - Sigmoid activation for multi-label output

## 2. Custom CNN Classifier (`custom_cnn_classifier.py`)

A custom CNN architecture built from scratch without using pre-trained models.

### Key Features:

#### Network Architecture
```python
def build_custom_cnn(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=7):
    model = Sequential([
        # First Block
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        # ... more layers
    ])
```
- 4 convolutional blocks with increasing filters (32→64→128→256)
- BatchNormalization and LeakyReLU for better training
- Dropout layers for regularization
- Global Average Pooling for dimension reduction

#### Advanced Data Generator
```python
class AdvancedDataGenerator(tf.keras.utils.Sequence):
    # Custom data augmentation and batch generation
```
- Real-time data augmentation
- Custom transformations:
  - Random horizontal flips
  - Rotation (-15° to 15°)
  - Brightness/contrast adjustments
- Efficient memory usage with batch processing

#### Training Features
- Custom F1 score metric
- Learning rate scheduling
- Early stopping with best weights restoration
- Model checkpointing

## 3. CUDA-Optimized Classifier (`cuda_optimized_classifier.py`)

GPU-accelerated implementation optimized for CUDA.

### Optimizations:

#### GPU Memory Management
```python
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
```
- Dynamic GPU memory allocation
- Mixed precision training (FP16)
- Optimized batch sizes for GPU

#### Efficient Data Pipeline
```python
def create_dataset(df, x_col, y_col, batch_size, is_training=True):
    # tf.data pipeline implementation
```
- Uses `tf.data` API for GPU-accelerated data processing
- Features:
  - Prefetching
  - Parallel processing
  - Memory-efficient data loading
  - GPU-accelerated augmentation

#### Custom F1 Score Implementation
```python
class F1Score(tf.keras.metrics.Metric):
    # GPU-optimized F1 score calculation
```
- Efficient metric computation on GPU
- Proper memory management
- Optimized for mixed precision training

### Training Pipeline
- Gradient accumulation
- Mixed precision optimizer
- GPU-accelerated data augmentation
- Efficient memory usage

## Usage Instructions

### 1. Prerequisites
```bash
pip install tensorflow-gpu numpy pandas opencv-python albumentations
```

### 2. Basic Usage
```python
# For improved classifier
from improved_clothing_classifier import build_model, CustomDataGenerator

# For custom CNN
from custom_cnn_classifier import build_custom_cnn, AdvancedDataGenerator

# For CUDA-optimized version
from cuda_optimized_classifier import build_cuda_optimized_cnn, create_dataset
```

### 3. Training Example
```python
# Prepare your data
label_columns = ['Hoodie', 'Kaos', 'biru', 'hitam', 'kuning', 'merah', 'putih']

# Choose your implementation
model = build_cuda_optimized_cnn(num_classes=len(label_columns))

# Create datasets
train_dataset = create_dataset(
    traindf,
    x_col='path',
    y_col=label_columns,
    batch_size=32,
    is_training=True
)

# Train the model
history = train_model_with_cuda(model, train_dataset, val_dataset)
```

## Performance Comparison

1. **Improved Clothing Classifier**
   - Advantages: Pre-trained features, sophisticated augmentation
   - Best for: Transfer learning, limited data scenarios

2. **Custom CNN Classifier**
   - Advantages: Full control over architecture, no pre-trained dependencies
   - Best for: Learning from scratch, custom requirements

3. **CUDA-Optimized Classifier**
   - Advantages: Fastest training, GPU optimization
   - Best for: Large datasets, production environments

## Best Practices

1. **Data Preparation**
   - Balance your dataset
   - Use appropriate image resolutions
   - Implement proper validation split

2. **Training**
   - Monitor GPU memory usage
   - Use appropriate batch sizes
   - Implement early stopping
   - Save best models

3. **Optimization**
   - Use mixed precision when possible
   - Enable GPU memory growth
   - Implement proper data pipelines

## Troubleshooting

1. **Memory Issues**
   - Reduce batch size
   - Enable memory growth
   - Use mixed precision training

2. **Performance Issues**
   - Check GPU utilization
   - Optimize data pipeline
   - Adjust learning rate

3. **Training Issues**
   - Monitor validation metrics
   - Adjust model architecture
   - Implement proper regularization

## Contributing

Feel free to submit issues and enhancement requests!
