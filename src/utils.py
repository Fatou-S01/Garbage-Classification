import matplotlib.pyplot as plt
import seaborn as sns
# Utiliser tensorflow.keras au lieu de keras directement
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def plot_confusion_matrix(conf_matrix, title):
    """ Plot the confusion matrix as a heatmap. """
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
# Initialisation du générateur d'augmentation d'images
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)