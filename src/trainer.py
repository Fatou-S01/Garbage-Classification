import sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import os
from PIL import Image
from sklearn.utils import shuffle
sys.path.append(str(Path.cwd().parent))
from src.make_dataset import load_data
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from src.utils import *



# def resize_images(images, target_size=(128, 128)):
#     resized_images = []
#     for img in images:
#         img = Image.fromarray((img * 255).astype(np.uint8)).resize(target_size)
#         img_array = np.array(img) / 255.0
#         resized_images.append(img_array)
#     return np.array(resized_images)

def resize_images(images, target_size=(128, 128)):
    """Resize images to the target size using cv2."""
    resized_images = []
    for img in images:
        img_resized = cv2.resize(img, target_size)
        resized_images.append(img_resized)
    return np.array(resized_images)

class ImageTrainer:
    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 model,
                 target_size=(128, 128),
                 batch_size=32,
                 epochs=20,
                 validation_size=0.2,
                 test_size=0.1,
                 random_state=42):
        self.data = data
        self.labels = labels
        self.model = model
        self.target_size = target_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_size = validation_size
        self.test_size = test_size
        self.random_state = random_state

        # Convert labels to integer
        self.label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
        self.labels = np.array([self.label_mapping[label] for label in labels])

        # Load and preprocess data
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        # First split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(self.data, self.labels, test_size=self.test_size, random_state=self.random_state)

        # Then split train+val into train and val
        val_size_adjusted = self.validation_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_adjusted, random_state=self.random_state)

        # Convert labels to categorical
        y_train = to_categorical(y_train, num_classes=len(self.label_mapping))
        y_val = to_categorical(y_val, num_classes=len(self.label_mapping))
        y_test = to_categorical(y_test, num_classes=len(self.label_mapping))

        # Resize images
        X_train = resize_images(X_train, target_size=self.target_size)
        X_val = resize_images(X_val, target_size=self.target_size)
        X_test = resize_images(X_test, target_size=self.target_size)

        # Create directories
        data_split_dir = os.path.join(os.path.dirname(os.getcwd()), 'data_split')
        train_dir = os.path.join(data_split_dir, 'train')
        val_dir = os.path.join(data_split_dir, 'val')
        test_dir = os.path.join(data_split_dir, 'test')

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Save images
        save_images(X_train, y_train, train_dir)
        save_images(X_val, y_val, val_dir)
        save_images(X_test, y_test, test_dir)

        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def eval_metrics(y_actual: np.ndarray, y_pred: np.ndarray, class_names: list) -> dict:
        """ Compute evaluation metrics

        Args:
            y_actual: Ground truth (correct) target values
            y_pred: Estimated target values.
            class_names: List of class names.

        Returns:
            dict: dictionary of evaluation metrics.
                Expected keys are: "accuracy", "classification_report", "confusion_matrix"
        """
        # Convert one-hot encoded y_actual to labels if necessary
        if len(y_actual.shape) > 1 and y_actual.shape[1] > 1:
            y_actual = np.argmax(y_actual, axis=1)
        
        # Ensure y_pred is also in label format
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)

        accuracy = accuracy_score(y_actual, y_pred)
        class_report_dict = classification_report(
            y_actual, y_pred, target_names=class_names, output_dict=True, labels=np.arange(len(class_names)), zero_division=0
        )
        class_report_df = pd.DataFrame(class_report_dict).transpose()
        conf_matrix = confusion_matrix(y_actual, y_pred, labels=np.arange(len(class_names)))
        
        return {
            "accuracy": accuracy,
            "classification_report": class_report_df,
            "confusion_matrix": conf_matrix
        }

    def train(self):
        # Create ImageDataGenerator for data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Create ImageDataGenerator for validation data
        val_test_datagen = ImageDataGenerator()

        # Fit the model using the augmented data generator
        train_generator = train_datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size)
        val_generator = val_test_datagen.flow(self.x_val, self.y_val, batch_size=self.batch_size)

        # Fit the model
        self.model.fit(
            train_generator,
            steps_per_epoch=len(self.x_train) // self.batch_size,
            epochs=self.epochs,
            validation_data=val_generator,
            validation_steps=len(self.x_val) // self.batch_size
        )

        # Evaluate Metrics on Training Data
        y_train_pred = self.model.predict(self.x_train)
        y_val_pred = self.model.predict(self.x_val)
        y_test_pred = self.model.predict(self.x_test)
        
        class_names = list(self.label_mapping.keys())
        
        train_metrics = self.eval_metrics(self.y_train, y_train_pred, class_names)
        val_metrics = self.eval_metrics(self.y_val, y_val_pred, class_names)
        test_metrics = self.eval_metrics(self.y_test, y_test_pred, class_names)

        logger.info(f"Train metrics: {train_metrics}")
        logger.info(f"Validation metrics: {val_metrics}")
        logger.info(f"Test metrics: {test_metrics}")

        return train_metrics, val_metrics, test_metrics, self.model
