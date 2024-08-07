import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.keras
import pandas as pd
import os
import cv2
import numpy as np

def plot_confusion_matrix(conf_matrix, title):
    """ Plot the confusion matrix as a heatmap. """
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def save_best_model_by_val_accuracy(experiment_name, save_directory):
    # Récupérer l'expérience par son nom
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found")

    experiment_id = experiment.experiment_id

    # Lister tous les runs de l'expérience
    runs = mlflow.search_runs(experiment_id)

    # Vérifier qu'il y a des runs
    if runs.empty:
        raise ValueError(f"No runs found for experiment {experiment_name}")

    # Trouver le run avec la meilleure précision de validation
    best_run = runs.loc[runs['metrics.val_accuracy'].idxmax()]

    # Afficher les informations du meilleur run
    print(f"Best run ID: {best_run.run_id}")
    print(f"Validation Accuracy: {best_run['metrics.val_accuracy']}")

    # Charger le modèle du meilleur run
    best_run_id = best_run.run_id
    model_name = best_run['params.model_name']  
    model_uri = f"runs:/{best_run_id}/model_{model_name}"
    model = mlflow.keras.load_model(model_uri)

    # Créer le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(save_directory, exist_ok=True)
    save_path = os.path.join(save_directory, f"best_model.h5")

    # Enregistrer le modèle localement
    model.save(save_path)
    print(f"Best model saved to {save_path}")

#Save images to the specified directory structure.
def save_images(images, labels, base_path):
    for i, (img, label) in enumerate(zip(images, labels)):
        label_dir = os.path.join(base_path, str(np.argmax(label)))
        os.makedirs(label_dir, exist_ok=True)
        img_path = os.path.join(label_dir, f'image_{i}.png')
        cv2.imwrite(img_path, img)