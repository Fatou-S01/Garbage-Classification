# Modèle de Classification des Déchets

Dans un monde confronté à des défis environnementaux croissants, la gestion efficace des déchets est devenue une nécessité impérieuse. La classification précise des déchets joue un rôle crucial dans ce processus, permettant d'améliorer significativement le recyclage, de réduire la pollution, et de promouvoir une économie circulaire. Elle contribue non seulement à la préservation de nos ressources naturelles, mais aussi à la sensibilisation du public sur l'importance du tri. Dans ce contexte, l'utilisation de technologies avancées comme l'intelligence artificielle pour la classification automatique des déchets représente une avancée majeure, offrant des solutions innovantes pour relever ces défis environnementaux pressants.

## Description
Ce projet implémente un modèle de classification des déchets à partir d'images. Il utilise l'apprentissage profond pour catégoriser automatiquement les déchets en 6 classes distinctes :

- Carton (cardboard)
- Verre (glass)
- Métal (metal)
- Papier (paper)
- Plastique (plastic)
- Déchets non recyclables (trash)

## Fonctionnalités
- Classification d'images de déchets en 6 catégories
- Interface utilisateur simple pour télécharger et classifier des images
- Statistiques sur les prédictions du modèle


## Données
Le modèle a été entraîné sur un ensemble de données comprenant des images de déchets dans les 6 catégories mentionnées. Elles proviennent du dataset Garbagge Classification provenant de Kaggle

## Prérequis
- Python 3.10.4
- TensorFlow
- Keras
- scikit-learn
- pandas
- numpy
- matplotlib
- loguru
- Kaggle API

## Installation
1. Clonez le dépôt du projet :
    ```bash
    git clone https://github.com/votre_nom_d_utilisateur/garbage_classification.git
    cd garbage_classification
    ```

2. Créez et activez un environnement virtuel :
    ```bash
    python -m venv original_venv
    source original_venv/Scripts/activate  # Pour Windows
    source original_venv/bin/activate      # Pour MacOS/Linux
    ```

3. Installez les dépendances requises :
    ```bash
    pip install -r requirements.txt
    ```

4. Configurez la clé API de Kaggle :
    - Obtenez votre fichier `kaggle.json` à partir de votre compte Kaggle.
    - Placez-le dans le répertoire `~/.kaggle/` (Linux/MacOS) ou `%USERPROFILE%\.kaggle\` (Windows).
      

## Tests
1. Pour exécuter les tests, utilisez `pytest` :
    ```bash
    pytest tests/ --disable-warnings
    ```


## Auteurs
- **Ndeye Fatou LAGNANE, Fatou SALL, Fama SARR** - *Initial Work* 
---

### Structure du Répertoire

```plaintext
garbage_classification/
├── API/
│   ├── app.py
│   ├── functions.py
│   ├── best_model.py
│
├── dashboard/
│
├── src/
│   ├── make_dataset.py
│   ├── utils.py
│   └── trainer.py
│
├── tests/
│   ├── conftest.py
│   ├── test_dataset.py
│   └── test_trainer.py
│
├── notebooks/
│   ├── data/
│   ├── data_split/
│   └── garbage/
│
├── Dockerfile/
│
├── requirements.txt
│
├── run_garbage_classification.sh
│
└── README.md
