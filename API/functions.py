import os

def save_base64_image(decoded_image, output_folder, output_filename):
    # Vérifie si le dossier de sortie existe, sinon le crée
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Crée le chemin complet du fichier de sortie
    output_path = os.path.join(output_folder, output_filename)
    
    # Écrit les données de l'image dans le fichier
    with open(output_path, 'wb') as output_file:
        output_file.write(decoded_image)
    

def predict():
    #Load modele et classification
    return "garbagge class"