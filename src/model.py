"""
Model architecture definition for Bangla Handwritten Character Recognition.
Built with Keras 3 (PyTorch backend).
"""

import os

# Set backend to torch before importing keras
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, models


def build_model(
    input_shape=(64, 64, 1),
    num_classes=84,
    dropout_rate=0.3
) -> keras.Model:
    """
    Build a deep convolutional neural network for 84-class Bangla character recognition.
    Features:
    - 4 Convolutional blocks with increasing filters (32 -> 64 -> 128 -> 256)
    - Batch Normalization and ReLU activation for stable gradient flow
    - MaxPooling and spatial regularization
    - Global Average Pooling + Dense layers with Softmax output
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Block 1: Low-level stroke edges (BatchNorm + ReLU, no early dropout for fast convergence)
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.ReLU(name="relu1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.ReLU(name="relu1_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2: Curvature and ligature components
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.ReLU(name="relu2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.ReLU(name="relu2_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3: Grapheme and compound patterns
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.ReLU(name="relu3_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.ReLU(name="relu3_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(dropout_rate * 0.7, name="drop3")(x)

    # Block 4: High-level semantic features
    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.ReLU(name="relu4_1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)
    x = layers.Dropout(dropout_rate, name="drop4")(x)

    # Classifier head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, use_bias=False, name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.ReLU(name="relu_dense")(x)
    x = layers.Dropout(dropout_rate, name="drop_dense")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="BanglaOCR_CNN")
    return model
