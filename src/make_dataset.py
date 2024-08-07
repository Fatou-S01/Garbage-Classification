import os
from kaggle.api.kaggle_api_extended import KaggleApi
import os
from PIL import Image
import cv2
from sklearn.utils import shuffle
from loguru import logger

#downloaded data from kaggle
def download_kaggle_dataset(dataset: str, download_path: str):
    """
    Downloads a dataset from Kaggle.

    Parameters:
    dataset (str): The Kaggle dataset identifier in the form "owner/dataset-name".
    download_path (str): The local path where the dataset should be saved.
    """
    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Ensure the download path exists
    if not os.path.exists:
        os.makedirs(download_path)
    
    # Download the dataset
    api.dataset_download_files(dataset, path=download_path, unzip=True)
    
    print(f"Dataset '{dataset}' downloaded to '{download_path}'.")


#read data from local image
def read_data(data_path):
    image_list = []
    label_list = []
    
    # Parcourir chaque dossier dans le chemin donné
    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        
        # Vérifier si c'est un dossier
        if os.path.isdir(class_path):
            # Parcourir chaque fichier dans le dossier de classe
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                
                # Lire l'image à partir du fichier
                image = cv2.imread(image_path)
                
                # Vérifier si l'image a été correctement lue
                if image is not None:
                    image_list.append(image)
                    label_list.append(class_name)
    image_list, label_list = shuffle(image_list, label_list, random_state=42)
    return image_list, label_list
  
  
  
#verification
def load_data(is_downloaded):
    dataset_name = 'asdasdasasdas/garbage-classification'
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    data_path = os.path.join(parent_directory, 'data','Garbage classification', 'Garbage classification')
    downloaded_path = os.path.join(parent_directory,'data')
    image_list = []
    label_list =  []
    
    if(is_downloaded):
      image_list, label_list = read_data(data_path)
    else:
      download_kaggle_dataset(dataset_name, downloaded_path)
      image_list, label_list = read_data(data_path)

    logger.info(f"Directories: {downloaded_path},{parent_directory}")
    logger.info(f"Dataset lo load: {data_path}")
    logger.info(f"Data shape: Images: {len(image_list)}, Labels: {len(label_list)}")
    
    return image_list, label_list