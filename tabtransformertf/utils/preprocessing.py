import math as m
import numpy as np
import pandas as pd
import tensorflow as tf
from tabtransformertf.utils.helper import corrupt_dataset
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


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


def build_numerical_prep(
    data: pd.DataFrame, numerical_features: list, y = None, n_bins: int = 10, type: str = "ple",
):
    if type not in ["ple"]:
        raise ValueError(f"Type {type} is not implemented yet")

    numerical_prep_layers = {}
    # Numerical prep layers
    for f in numerical_features:
        num_emb_layer = PLE(n_bins)
        num_emb_layer.adapt(data[f].astype(np.float32).values.reshape(-1, 1), y = y)
        numerical_prep_layers[f] = num_emb_layer

    return numerical_prep_layers


def df_to_pretrain_dataset(
    x: pd.DataFrame,
    numeric_columns: list,
    categorical_columns: list,
    shuffle: bool = True,
    batch_size: int = 512,
    p_replace: float = 0.3,
):
    x, y = corrupt_dataset(x[numeric_columns + categorical_columns], p_replace)
    x[numeric_columns] = x[numeric_columns].astype(float)
    x[categorical_columns] = x[categorical_columns].astype(str)

    dataset = {}
    for key, value in x.items():
        dataset[key] = value[:, tf.newaxis]

    dataset = tf.data.Dataset.from_tensor_slices((dict(dataset), y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(x))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset


class PLE(tf.keras.layers.Layer):
    def __init__(self, n_bins=10):
        super(PLE, self).__init__()
        self.n_bins = n_bins

    def adapt(self, data, y=None, task='classification', tree_params = {}):
        if y is not None:
            if task == 'classification':
                dt = DecisionTreeClassifier(max_leaf_nodes=self.n_bins, **tree_params)
            elif task == 'regression':
                dt = DecisionTreeRegressor(max_leaf_nodes=self.n_bins, **tree_params)
            else:
                raise ValueError("This task is not supported")
            dt.fit(data, y)
            bins = tf.sort(tf.cast(tf.unique(dt.tree_.threshold).y, dtype=tf.float32))
        else:
            interval = 1 / self.n_bins
            bins = tf.unique(
                [
                    tf.cast(np.quantile(data, q), tf.float32)
                    for q in np.arange(0.0, 1 + interval, interval)
                ]
            ).y

        self.n_bins = len(bins)
        init = tf.lookup.KeyValueTensorInitializer(
            [i for i in range(self.n_bins)], bins
        )
        self.lookup_table = tf.lookup.StaticHashTable(init, default_value=-1)
        self.lookup_size = self.lookup_table.size()

    def call(self, x):
        ple_encoding_one = tf.ones((tf.shape(x)[0], self.n_bins))
        ple_encoding_zero = tf.zeros((tf.shape(x)[0], self.n_bins))

        left_masks = tf.TensorArray(tf.bool, size=0, dynamic_size=True)
        right_masks = tf.TensorArray(tf.bool, size=0, dynamic_size=True)
        other_case = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        for i in range(1, self.n_bins + 1):
            i = tf.constant(i)
            left_mask = (x < self.lookup_table.lookup(i - 1)) & (i > 1)
            right_mask = (x >= self.lookup_table.lookup(i)) & (i < self.n_bins)
            v = (x - self.lookup_table.lookup(i - 1)) / (
                self.lookup_table.lookup(i) - self.lookup_table.lookup(i - 1)
            )
            left_masks = left_masks.write(left_masks.size(), left_mask)
            right_masks = right_masks.write(right_masks.size(), right_mask)
            other_case = other_case.write(other_case.size(), v)

        left_masks = tf.transpose(tf.squeeze(left_masks.stack()))
        right_masks = tf.transpose(tf.squeeze(right_masks.stack()))
        other_case = tf.transpose(tf.squeeze(other_case.stack()))

        other_mask = right_masks == left_masks  # both are false
        other_case = tf.cast(other_case, tf.float32)
        enc = tf.where(left_masks, ple_encoding_zero, ple_encoding_one)
        enc = tf.reshape(tf.where(other_mask, other_case, enc), (-1, 1, self.n_bins))

        return enc

class Periodic(tf.keras.layers.Layer):
  def __init__(self, emb_dim, n_bins=50, sigma=5):
      super(Periodic, self).__init__()
      self.n_bins = n_bins
      self.emb_dim = emb_dim
      self.sigma = sigma
  
  def build(self, input_shape):  # Create the state of the layer (weights)
    w_init = tf.random_normal_initializer(stddev=self.sigma)
    self.p = tf.Variable(
        initial_value=w_init(shape=(input_shape[-1], self.n_bins),
                             dtype='float32'),
        trainable=True)

    self.l = tf.Variable(
        initial_value=w_init(
            shape=(input_shape[-1], self.n_bins*2, self.emb_dim), dtype='float32' # features, n_bins, emb_dim
            ), trainable=True)

  def call(self, inputs):  # Defines the computation from inputs to outputs
    v = 2 * m.pi * self.p[None] * inputs[..., None]
    emb = tf.concat([tf.math.sin(v), tf.math.cos(v)], axis=-1)
    emb = tf.einsum('fne, bfn -> bfe', self.l, emb)
    emb = tf.nn.relu(emb)

    return emb
