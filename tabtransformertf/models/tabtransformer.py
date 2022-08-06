import tensorflow as tf
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)
from tabtransformertf.utils.helper import build_mlp


class TransformerBlock(Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        att_dropout: float = 0.1,
        ff_dropout: float = 0.1,
    ):
        """Transformer model for TabTransformer

        Args:
            embed_dim (int): embedding dimensions
            num_heads (int): number of attention heads
            ff_dim (int): size of feed-forward layer
            att_dropout (float, optional): dropout rate in multi-headed attention layer. Defaults to 0.1.
            ff_dropout (float, optional): dropout rate in feed-forward layer. Defaults to 0.1.
        """
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=att_dropout
        )
        self.skip1 = Add()
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation=gelu), Dropout(ff_dropout), Dense(embed_dim),]
        )
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.skip2 = Add()

    def call(self, inputs):
        # Multi headed attention
        attention_output = self.att(inputs, inputs)
        # Skip connection
        attention_output = self.skip1([inputs, attention_output])
        # Layer norm
        attention_output = self.layernorm1(attention_output)
        # Feed Forward
        feedforward_output = self.ffn(attention_output)
        # Skip connection
        feedforward_output = self.skip2([feedforward_output, attention_output])
        # Layer norm
        transformer_output = self.layernorm2(feedforward_output)

        return transformer_output


class TabTransformer(tf.keras.Model):
    def __init__(
        self,
        categorical_features: list,
        numerical_features: list,
        categorical_lookup: dict,
        out_dim: int,
        out_activation: str,
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        mlp_hidden_factors: list = [2, 4],
        numerical_discretisers: dict = None,
        use_column_embedding: bool = True,
    ):
        """TabTransformer Tensorflow Model

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
            numerical_discretisers (dict, optional): dictionary with numerical feature names as keys and adapted Discretizer and IntegerLookup layers as values. Defaults to None.
            use_column_embedding (bool, optional): flag to use fixed column positional embeddings. Defaults to True.
        """

        super(TabTransformer, self).__init__()
        self.numerical = numerical_features
        self.categorical = categorical_features
        self.quantize = numerical_discretisers is not None
        self.num_categories = [
            categorical_lookup[c].vocabulary_size() for c in self.categorical
        ]

        # ---------- Numerical Input -----------
        if len(self.numerical) > 0:
            # If we want to quantise numeric features
            if self.quantize:
                self.num_bins = [
                    numerical_discretisers[n][1].vocabulary_size()
                    for n in self.numerical
                ]
                # Discretisation layers
                self.numerical_discretisers = [
                    numerical_discretisers[n][0] for n in self.numerical
                ]
                # Lookup layers
                self.numerical_lookup = [
                    numerical_discretisers[n][1] for n in self.numerical
                ]
                # Embedding layers
                self.num_embedding_layers = []
                for bins in self.num_bins:
                    numerical_embedding = Embedding(
                        input_dim=bins, output_dim=embedding_dim
                    )
                    self.num_embedding_layers.append(numerical_embedding)
            else:
                # If not quantising, then simply normalise and concatenate
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
            if self.quantize:
                num_columns += len(self.numerical)
            self.column_embedding = Embedding(
                input_dim=num_columns, output_dim=embedding_dim
            )
            self.column_indices = tf.range(start=0, limit=num_columns, delta=1)

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
                )
            )
        self.flatten_transformer_output = Flatten()

        # MLP
        self.pre_mlp_concatenation = Concatenate()

        # mlp layers
        if self.quantize:
            mlp_input_dim = embedding_dim * (
                len(self.numerical) + len(self.categorical)
            )
        else:
            mlp_input_dim = len(self.numerical) + embedding_dim * len(self.categorical)

        self.mlp_final = build_mlp(mlp_input_dim, mlp_hidden_factors, ff_dropout)
        self.output_layer = Dense(out_dim, activation=out_activation)

    def call(self, inputs):
        numerical_feature_list = []
        categorical_feature_list = []

        if len(self.numerical) > 0:
            # Each numeric feature needs to be binned, looked up, and embedded
            for i, n in enumerate(self.numerical):
                if self.quantize:
                    num_binned = self.numerical_discretisers[i](inputs[n])
                    num_binned = self.numerical_lookup[i](num_binned)
                    num_embedded = self.num_embedding_layers[i](num_binned)
                    numerical_feature_list.append(num_embedded)
                else:
                    # Otherwise we pass it as it is
                    numerical_feature_list.append(inputs[n])

        for i, c in enumerate(self.categorical):
            cat_encoded = self.categorical_lookups[i](inputs[c])
            cat_embedded = self.cat_embedding_layers[i](cat_encoded)
            categorical_feature_list.append(cat_embedded)

        if self.quantize:
            # Stack categorical embeddings for the Tansformer.
            transformer_inputs = self.embedded_concatenation(
                numerical_feature_list + categorical_feature_list
            )
        else:
            transformer_inputs = self.embedded_concatenation(categorical_feature_list)

        if self.use_column_embedding:
            # Add column embeddings
            transformer_inputs += self.column_embedding(self.column_indices)

        for transformer in self.transformers:
            transformer_inputs = transformer(transformer_inputs)

        # Flatten the "contextualized" embeddings of the features.
        mlp_input = self.flatten_transformer_output(transformer_inputs)

        # In case we don't quantize, we want to normalise and concatenate numerical features with embeddings
        if (self.quantize is False) and (len(self.numerical) > 0):
            numerical_inputs = self.numerical_concatenation(numerical_feature_list)
            numerical_inputs = self.continuous_normalization(numerical_inputs)
            mlp_input = self.pre_mlp_concatenation([mlp_input, numerical_inputs])

        # Pass through MLP
        mlp_output = self.mlp_final(mlp_input)
        output = self.output_layer(mlp_output)

        return output
