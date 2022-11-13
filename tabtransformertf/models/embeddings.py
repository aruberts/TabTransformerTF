import tensorflow as tf
import numpy as np
import math as m
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


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


class NEmbedding(tf.keras.Model):
    def __init__(
        self,
        feature_names: list,
        X: np.array,
        y: np.array = None,
        task: str = None,
        emb_dim: int = 32,
        emb_type: str = 'linear',
        n_bins: int = 10,
        sigma: float = 1,
        tree_params = {},
    ):
        super(NEmbedding, self).__init__()

        if emb_type not in ['linear', 'ple', 'periodic']:
            raise ValueError("This emb_type is not supported")
        
        self.num_features = len(feature_names)
        self.features = feature_names
        self.emb_type = emb_type
        self.emb_dim = emb_dim
        
        # Initialise embedding layers
        if emb_type == 'ple':
            self.embedding_layers = {}
            self.linear_layers = {}
            for i, f in enumerate(feature_names):
                emb_l = PLE(n_bins)
                if y is None:
                    emb_l.adapt(X[:, i], tree_params=tree_params)
                else:
                    emb_l.adapt(X[:, i].reshape(-1, 1), y, task=task, tree_params=tree_params)

                lin_l = tf.keras.layers.Dense(emb_dim, activation='relu')
                
                self.embedding_layers[f] = emb_l
                self.linear_layers[f] = lin_l

        elif emb_type == 'periodic':
            # There's just 1 periodic layer
            self.embedding_layer = Periodic(
                n_bins = n_bins,
                emb_dim = emb_dim,
                sigma = sigma)
        else:
            # Initialise linear layer
            w_init = tf.random_normal_initializer()
            self.linear_w = tf.Variable(
                initial_value=w_init(
                    shape=(self.num_features, 1, self.emb_dim), dtype='float32' # features, n_bins, emb_dim
                ), trainable=True)
            self.linear_b = tf.Variable(
                w_init(
                    shape=(self.num_features, 1), dtype='float32' # features, n_bins, emb_dim
                ), trainable=True)
    
    
    def embed_column(self, f, data):
        emb = self.linear_layers[f](self.embedding_layers[f](data))
        return emb
   
    def call(self, x):
        if self.emb_type == 'ple':
            emb_columns = []
            for i, f in enumerate(self.features):
                emb_columns.append(self.embed_column(f, x[:, i]))
            embs = tf.concat(emb_columns, axis=1)
            
        elif self.emb_type == 'periodic':
            embs = self.embedding_layer(x)
        else:
            embs = tf.einsum('f n e, b f -> bfe', self.linear_w, x)
            embs = tf.nn.relu(embs + self.linear_b)
            
        return embs

class CEmbedding(tf.keras.Model):
    def __init__(
        self,
        feature_names: list,
        X: np.array,
        emb_dim: int = 32,
    ):
        super(CEmbedding, self).__init__()
        self.features = feature_names
        self.emb_dim = emb_dim
        
        self.category_prep_layers = {}
        self.emb_layers = {}
        for i, c in enumerate(self.features):
            lookup = tf.keras.layers.StringLookup(vocabulary=list(np.unique(X[:, i])))
            emb = tf.keras.layers.Embedding(input_dim=lookup.vocabulary_size(), output_dim=self.emb_dim)

            self.category_prep_layers[c] = lookup
            self.emb_layers[c] = emb
    
    def embed_column(self, f, data):
        return self.emb_layers[f](self.category_prep_layers[f](data))

    def call(self, x):
        emb_columns = []
        for i, f in enumerate(self.features):
            emb_columns.append(self.embed_column(f, x[:, i]))
        
        embs = tf.stack(emb_columns, axis=1)
        return embs