import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.keras
import pandas as pd
import os

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
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.keras.load_model(model_uri)

    # Créer le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(save_directory, exist_ok=True)
    save_path = os.path.join(save_directory, "best_model.h5")

    # Enregistrer le modèle localement
    model.save(save_path)
    print(f"Best model saved to {save_path}")
