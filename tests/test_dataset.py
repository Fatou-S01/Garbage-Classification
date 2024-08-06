import pytest
import numpy as np
from make_dataset import load_data

def test_load_data():
    images, labels = load_data()
    assert isinstance(images, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert len(images) == len(labels)
    assert images.ndim == 4  # Assurez-vous que les images ont 4 dimensions
    assert labels.ndim == 1  # Assurez-vous que les labels ont 1 dimension