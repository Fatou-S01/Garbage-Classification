import pytest
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from src.trainer import ImageTrainer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def create_dummy_cnn_model(input_shape=(128, 128, 3), num_classes=6):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def test_image_trainer():
    images = np.random.rand(100, 128, 128, 3)
    labels = np.random.randint(0, 6, 100)
    model = create_dummy_cnn_model()
    trainer = ImageTrainer(data=images, labels=labels, model=model, epochs=1)
    train_metrics, val_metrics, test_metrics, trained_model = trainer.train()
    assert 'accuracy' in train_metrics
    assert 'accuracy' in val_metrics
    assert 'accuracy' in test_metrics
    assert train_metrics['accuracy'] > 0
    assert val_metrics['accuracy'] > 0
    assert test_metrics['accuracy'] > 0
