import pytest
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2
from src.trainer import ImageTrainer

# Dummy CNN model for testing
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Test Initialization
def test_image_trainer_initialization():
    images = np.random.rand(100, 128, 128, 3)
    labels = np.random.randint(0, 6, 100)
    model = create_dummy_cnn_model()
    trainer = ImageTrainer(data=images, labels=labels, model=model)

    assert trainer.data.shape == (100, 128, 128, 3)
    assert trainer.labels.shape == (100,)
    assert len(trainer.label_mapping) == 6

# Test Data Loading and Preprocessing
def test_load_and_preprocess_data():
    images = np.random.rand(100, 128, 128, 3)
    labels = np.random.randint(0, 6, 100)
    model = create_dummy_cnn_model()
    trainer = ImageTrainer(data=images, labels=labels, model=model)

    X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_and_preprocess_data()

    assert X_train.shape[1:] == (128, 128, 3)
    assert X_val.shape[1:] == (128, 128, 3)
    assert X_test.shape[1:] == (128, 128, 3)
    assert y_train.shape[1] == 6
    assert y_val.shape[1] == 6
    assert y_test.shape[1] == 6


# Test Metrics Evaluation
def test_eval_metrics():
    y_actual = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 2, 1, 0])
    class_names = ['class0', 'class1', 'class2']

    metrics = ImageTrainer.eval_metrics(y_actual, y_pred, class_names)

    assert 'accuracy' in metrics
    assert 'classification_report' in metrics
    assert 'confusion_matrix' in metrics


