import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.activations import selu

def build_mlp(input_dim, factors, dropout):
    hidden_units = [input_dim // f for f in factors]

    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(BatchNormalization()),
        mlp_layers.append(Dense(units, activation=selu))
        mlp_layers.append(Dropout(dropout))

    return tf.keras.Sequential(mlp_layers)