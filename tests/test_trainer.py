import pytest
import numpy as np
from src.trainer import ImageTrainer, create_cnn_model

def test_image_trainer():
    images = np.random.rand(100, 128, 128, 3)
    labels = np.random.randint(0, 6, 100)
    model = create_cnn_model(input_shape=(128, 128, 3), num_classes=6)
    trainer = ImageTrainer(data=images, labels=labels, model=model, epochs=1)
    train_metrics, val_metrics, test_metrics, trained_model = trainer.train()
    assert 'accuracy' in train_metrics
    assert 'accuracy' in val_metrics
    assert 'accuracy' in test_metrics