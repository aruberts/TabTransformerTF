import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import selu
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
import matplotlib.pyplot as plt

def build_mlp(input_dim, factors, dropout):
    hidden_units = [input_dim // f for f in factors]

    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(BatchNormalization()),
        mlp_layers.append(Dense(units, activation=selu))
        mlp_layers.append(Dropout(dropout))

    return tf.keras.Sequential(mlp_layers)


def generate_mask(x, p_replace=0.2):
    m = np.random.choice([False, True], size=x.shape, p=[1-p_replace, p_replace])
    return m


def corrupt_dataset(x, p_replace=0.2):
    mask = generate_mask(x, p_replace=p_replace)
    shuffled_data = np.random.permutation(x)
    corrupted_data = x.copy()
    corrupted_data = corrupted_data.mask(mask, shuffled_data)
    new_mask = (corrupted_data == x).values
    return corrupted_data, new_mask


def get_model_importances(importances, title="Importances"):
    imps_sorted = importances.mean().sort_values(ascending=False)
    
    plt.figure(figsize=(15,7))
    ax = imps_sorted.plot.bar()
    for p in ax.patches:
        ax.annotate(str(np.round(p.get_height(), 4)), (p.get_x(), p.get_height() * 1.01))
    plt.title(title)
    plt.show()
    
    return imps_sorted
