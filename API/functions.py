import os
from tensorflow.keras.models import Model, load_model
import cv2
import numpy as np
import glob


def save_base64_image(decoded_image, output_folder, output_filename):
    # Vérifie si le dossier de sortie existe, sinon le crée
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Crée le chemin complet du fichier de sortie
    output_path = os.path.join(output_folder, output_filename)
    
    # Écrit les données de l'image dans le fichier
    with open(output_path, 'wb') as output_file:
        output_file.write(decoded_image)
    


def resize_images(images, target_size=(128, 128)):
    """Resize images to the target size using cv2."""
    resized_images = []
    for img in images:
        img_resized = cv2.resize(img, target_size)
        resized_images.append(img_resized)
    return np.array(resized_images)

def predict_image():
    # Chemin vers le modèle sauvegardé
    project_dir = os.path.dirname(os.getcwd())
    parent_dir = os.path.join(project_dir,'API')
    model_path = os.path.join(parent_dir,'best_model.h5')
    image_path = os.path.join(parent_dir,'garbaggeImage','garbaggePhoto.png')
    img_array = cv2.imread(image_path)
    image = resize_images([img_array])
    # Charger le modèle
    model = load_model(model_path)
    predictions = model.predict(image)
    category = np.argmax(predictions, axis=1)
    categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    predicted_category = categories[category[0]]
    
    return predicted_category


