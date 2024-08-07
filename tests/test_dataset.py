import os
import pytest
import cv2
import numpy as np
from unittest import mock
from unittest.mock import patch, MagicMock
from src.make_dataset import download_kaggle_dataset, read_data, load_data

# Test for download_kaggle_dataset
@patch('src.make_dataset.KaggleApi')
def test_download_kaggle_dataset(mock_kaggle_api):
    dataset_name = "owner/dataset-name"
    download_path = "./data"
    
    mock_api_instance = mock_kaggle_api.return_value
    mock_api_instance.dataset_download_files.return_value = None
    
    download_kaggle_dataset(dataset_name, download_path)
    
    mock_api_instance.authenticate.assert_called_once()
    mock_api_instance.dataset_download_files.assert_called_once_with(dataset_name, path=download_path, unzip=True)

# Test for read_data
def test_read_data():
    # Setup
    data_path = "./test_data"
    os.makedirs(data_path, exist_ok=True)
    class_name = "class1"
    class_path = os.path.join(data_path, class_name)
    os.makedirs(class_path, exist_ok=True)
    
    # Create dummy images
    for i in range(5):
        img_path = os.path.join(class_path, f"image_{i}.png")
        cv2.imwrite(img_path, (255 * np.random.rand(100, 100, 3)).astype(np.uint8))
    
    # Test
    images, labels = read_data(data_path)
    
    # Assertions
    assert len(images) == 5
    assert len(labels) == 5
    assert all(label == class_name for label in labels)
    
    # Cleanup
    import shutil
    shutil.rmtree(data_path)

# Test for load_data
@patch('src.make_dataset.download_kaggle_dataset')
@patch('src.make_dataset.read_data')
def test_load_data(mock_read_data, mock_download_kaggle_dataset):
    dataset_name = 'asdasdasasdas/garbage-classification'
    is_downloaded = True
    
    mock_read_data.return_value = (['image1', 'image2'], ['label1', 'label2'])
    
    images, labels = load_data(is_downloaded)
    
    if is_downloaded:
        mock_read_data.assert_called_once()
        mock_download_kaggle_dataset.assert_not_called()
    else:
        mock_download_kaggle_dataset.assert_called_once_with(dataset_name, mock.ANY)
        mock_read_data.assert_called_once()
    
    assert len(images) == 2
    assert len(labels) == 2
    assert images == ['image1', 'image2']
    assert labels == ['label1', 'label2']