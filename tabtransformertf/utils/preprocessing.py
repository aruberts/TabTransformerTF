from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np


def df_to_dataset(
    dataframe: pd.DataFrame,
    target: str = None,
    shuffle: bool = True,
    batch_size: int = 512,
):
    df = dataframe.copy()
    if target:
        labels = df.pop(target)
        dataset = {}
        for key, value in df.items():
            dataset[key] = value[:, tf.newaxis]

        dataset = tf.data.Dataset.from_tensor_slices((dict(dataset), labels))
    else:
        dataset = {}
        for key, value in df.items():
            dataset[key] = value[:, tf.newaxis]

        dataset = tf.data.Dataset.from_tensor_slices(dict(dataset))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset


def build_categorical_prep(data: pd.DataFrame, categorical_features: list):
    category_prep_layers = {}
    for c in tqdm(categorical_features):
        lookup = tf.keras.layers.StringLookup(vocabulary=data[c].unique())
        category_prep_layers[c] = lookup

    return category_prep_layers


def build_numerical_prep(data: pd.DataFrame, numerical_features: list, qs: int = 10):
    numeric_prep_layers = {}
    for n in tqdm(numerical_features):
        _, bin_bound = pd.qcut(data[n], qs, retbins=True, duplicates="drop")
        discretiser = tf.keras.layers.Discretization(bin_boundaries=bin_bound)
        discretised_column = discretiser(data[n].values)
        int_lookup = tf.keras.layers.IntegerLookup(
            vocabulary=np.unique(discretised_column.numpy())
        )
        numeric_prep_layers[n] = (discretiser, int_lookup)

    return numeric_prep_layers
