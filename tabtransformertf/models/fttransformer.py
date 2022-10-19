import tensorflow as tf
from tabtransformertf.models.tabtransformer import TransformerBlock
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Embedding,
    Flatten,
    LayerNormalization,
)


class FTTransformerEncoder(tf.keras.Model):
    def __init__(
        self,
        categorical_features: list,
        numerical_features: list,
        categorical_lookup: dict,
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        numerical_embedding_type: str = 'linear',
        numerical_embeddings: dict = None,
        use_column_embedding: bool = True,
        explainable=False,
    ):
        """FTTransformer Encoder
        Args:
            categorical_features (list): names of categorical features
            numerical_features (list): names of numeric features
            categorical_lookup (dict): dictionary with categorical feature names as keys and adapted StringLookup layers as values
            out_dim (int): model output dimensions
            out_activation (str): model output activation
            embedding_dim (int, optional): embedding dimensions. Defaults to 32.
            depth (int, optional): number of transformer blocks. Defaults to 4.
            heads (int, optional): number of attention heads. Defaults to 8.
            attn_dropout (float, optional): dropout rate in transformer. Defaults to 0.1.
            ff_dropout (float, optional): dropout rate in mlps. Defaults to 0.1.
            mlp_hidden_factors (list[int], optional): numbers by which we divide dimensionality. Defaults to [2, 4].
            numerical_embeddings (dict, optional): dictionary with numerical feature names as keys and adapted numerical embedding layers as values. Defaults to None.
            numerical_embedding_type (str, optional): name of the numerical embedding procedure. Defaults to linear.
            use_column_embedding (bool, optional): flag to use fixed column positional embeddings. Defaults to True.
            explainable (bool, optional): flag to output importances inferred from attention weights. Defaults to False.
        """

        super(FTTransformerEncoder, self).__init__()
        self.numerical = numerical_features
        self.categorical = categorical_features
        self.numerical_embedding_type = numerical_embedding_type
        self.embedding_dim = embedding_dim
        self.explainable = explainable
        self.depth = depth
        self.heads = heads

        self.num_categories = [
            categorical_lookup[c].vocabulary_size() for c in self.categorical
        ]
        if numerical_embedding_type not in {None, "linear", "ple"}:
            raise ValueError("numerical_embedding_type must be linear or ple")

        if (numerical_embedding_type == "ple") & (numerical_embeddings is None):
            raise ValueError(
                "When embedding type is PLE, numerical_embeddings must be a dict with PLE layers"
            )

        # ---------- Numerical Input -----------
        if len(self.numerical) > 0:
            # If we want to embed numerical features
            if self.numerical_embedding_type:
                # Layers to transform numeric into embedding
                self.numerical_embeddings = numerical_embeddings
                # Linear layer after embedding
                self.numerical_embedding_linear = [
                    Dense(embedding_dim, activation="relu") for n in self.numerical
                ]
            else:
                # If not embedding, then simply normalise and concatenate
                self.continuous_normalization = LayerNormalization()
                self.numerical_concatenation = Concatenate(axis=1)

        # ---------- Categorical Input -----------

        # String lookups for categorical
        self.categorical_lookups = [categorical_lookup[c] for c in self.categorical]

        # Categorical input embedding
        self.cat_embedding_layers = []
        for number_of_classes in self.num_categories:
            category_embedding = Embedding(
                input_dim=number_of_classes, output_dim=embedding_dim
            )
            self.cat_embedding_layers.append(category_embedding)

        # Column embedding
        self.use_column_embedding = use_column_embedding
        if use_column_embedding:
            num_columns = len(self.categorical)
            if self.numerical_embedding_type:
                num_columns += len(self.numerical)
            self.column_embedding = Embedding(
                input_dim=num_columns + 1, output_dim=embedding_dim
            )
            self.column_indices = tf.range(start=0, limit=num_columns + 1, delta=1)

        # Embedding concatenation layer
        self.embedded_concatenation = Concatenate(axis=1)

        # adding transformers
        self.transformers = []
        for _ in range(depth):
            self.transformers.append(
                TransformerBlock(
                    embedding_dim,
                    heads,
                    embedding_dim,
                    att_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    explainable=self.explainable,
                )
            )
        self.flatten_transformer_output = Flatten()

        # MLP
        self.pre_mlp_concatenation = Concatenate()

        # CLS token
        w_init = tf.random_normal_initializer()
        self.cls_weights = tf.Variable(
            initial_value=w_init(shape=(1, embedding_dim), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        numerical_feature_list = []
        categorical_feature_list = []
        importances = []

        # Each numeric feature needs to be binned, looked up, and embedded
        for i, n in enumerate(self.numerical):
            if self.numerical_embedding_type == "ple":
                num_embedded = self.numerical_embeddings[n](inputs[n])
                num_embedded = self.numerical_embedding_linear[i](num_embedded)
                numerical_feature_list.append(num_embedded)
            elif self.numerical_embedding_type == "linear":
                num_embedded = tf.expand_dims(inputs[n], axis=1)
                num_embedded = self.numerical_embedding_linear[i](num_embedded)
                numerical_feature_list.append(num_embedded)
            else:
                raise ValueError("numerical_embedding_type must be 'ple' or 'linear'")

        for i, c in enumerate(self.categorical):
            cat_encoded = self.categorical_lookups[i](inputs[c])
            cat_embedded = self.cat_embedding_layers[i](cat_encoded)
            categorical_feature_list.append(cat_embedded)

        cls_tokens = tf.repeat(self.cls_weights, repeats=tf.shape(inputs[c])[0], axis=0)
        cls_tokens = tf.expand_dims(cls_tokens, axis=1)

        # Stack categorical embeddings for the Tansformer.
        transformer_inputs = self.embedded_concatenation(
            numerical_feature_list + categorical_feature_list + [cls_tokens]
        )

        if self.use_column_embedding:
            # Add column embeddings
            transformer_inputs += self.column_embedding(self.column_indices)

        for transformer in self.transformers:
            if self.explainable:
                transformer_inputs, att_weights = transformer(transformer_inputs)
                importances.append(tf.reduce_sum(att_weights[:, :, -1, :], axis=1))
            else:
                transformer_inputs = transformer(transformer_inputs)

        if self.explainable:
            # Sum across the layers
            importances = tf.reduce_sum(tf.stack(importances), axis=0) / (
                self.depth * self.heads
            )
            return transformer_inputs, importances
        else:
            return transformer_inputs


class FTTransformer(tf.keras.Model):
    def __init__(
        self,
        out_dim: int,
        out_activation: str,
        final_layer_size: int = 32,
        categorical_features: list = None,
        numerical_features: list = None,
        categorical_lookup: dict = None,
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        numerical_embedding_type: str = None,
        numerical_embeddings: dict = None,
        use_column_embedding: bool = True,
        explainable=False,
        encoder=None,
    ):
        super(FTTransformer, self).__init__()

        # Initialise encoder
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = FTTransformerEncoder(
                categorical_features = categorical_features,
                numerical_features = numerical_features,
                categorical_lookup = categorical_lookup,
                embedding_dim = embedding_dim,
                depth = depth,
                heads = heads,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                numerical_embedding_type = numerical_embedding_type,
                numerical_embeddings = numerical_embeddings,
                use_column_embedding = use_column_embedding,
                explainable = explainable,
            )

        # mlp layers
        self.final_layer = Dense(final_layer_size, activation="relu")
        self.output_layer = Dense(out_dim, activation=out_activation)

    def call(self, inputs):
        if self.encoder.explainable:
            x, expl = self.encoder(inputs)
        else:
            x = self.encoder(inputs)
        x = self.final_layer(x[:, -1, :])
        output = self.output_layer(x)

        if self.encoder.explainable:
            # Explaianble models return two outputs
            return {"output": output, "importances": expl}
        else:
            return output
