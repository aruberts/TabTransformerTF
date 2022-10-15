import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import selu
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout


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
