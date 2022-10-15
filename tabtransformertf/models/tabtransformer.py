import tensorflow as tf
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
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


class TabTransformerEncoder(tf.keras.Model):
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
        numerical_embeddings: dict = None,
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

        super(TabTransformerEncoder, self).__init__()
        self.numerical = numerical_features
        self.categorical = categorical_features
        self.embed_numeric = numerical_embeddings is not None
        self.num_categories = [
            categorical_lookup[c].vocabulary_size() for c in self.categorical
        ]

        # ---------- Numerical Input -----------
        if len(self.numerical) > 0:
            # If we want to quantise numeric features
            if self.embed_numeric:
                # Layers to transform numeric into embedding
                self.numerical_embeddings = numerical_embeddings
                # Linear layer after embedding
                self.numerical_embedding_linear = [
                    Dense(embedding_dim, activation='relu') for n in self.numerical
                ]
                
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
            if self.embed_numeric:
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

    def call(self, inputs):
        numerical_feature_list = []
        categorical_feature_list = []

        if len(self.numerical) > 0:
            # Each numeric feature needs to be binned, looked up, and embedded
            for i, n in enumerate(self.numerical):
                if self.embed_numeric:
                    num_embedded = self.numerical_embeddings[n](inputs[n])
                    num_embedded = self.numerical_embedding_linear[i](num_embedded)
                    numerical_feature_list.append(num_embedded)
                else:
                    # Otherwise we pass it as it is
                    numerical_feature_list.append(inputs[n])

        for i, c in enumerate(self.categorical):
            cat_encoded = self.categorical_lookups[i](inputs[c])
            cat_embedded = self.cat_embedding_layers[i](cat_encoded)
            categorical_feature_list.append(cat_embedded)

        if self.embed_numeric:
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
        if (self.embed_numeric is False) and (len(self.numerical) > 0):
            numerical_inputs = self.numerical_concatenation(numerical_feature_list)
            numerical_inputs = self.continuous_normalization(numerical_inputs)
            mlp_input = self.pre_mlp_concatenation([mlp_input, numerical_inputs])

        return mlp_input

class TabTransformerRTD(tf.keras.Model):
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
        numerical_discretisers: dict = None,
        use_column_embedding: bool = True,
        rtd_factor=2,
    ):

        super(TabTransformerRTD, self).__init__()

        # Initialise encoder
        self.encoder = TabTransformerEncoder(
            categorical_features,
            numerical_features,
            categorical_lookup,
            embedding_dim,
            depth,
            heads,
            attn_dropout,
            ff_dropout,
            numerical_discretisers,
            use_column_embedding,
        )

        self.decoders = []
        self.n_features = len(categorical_features) + len(numerical_features)
        n_features_emb = len(categorical_features) + len(numerical_features) * embedding_dim
        for _ in range(self.n_features):
            decoder = tf.keras.Sequential([
                BatchNormalization(),
                Dense(n_features_emb // rtd_factor, activation='selu'),
                Dense(1, activation='sigmoid')
            ])
            self.decoders.append(decoder)
        self.concatenate_output = Concatenate(axis=1)
        
    def call(self, inputs):
        contextual_encoding = self.encoder(inputs)
        rtd_prediction = [self.decoders[i](contextual_encoding) for i in range(self.n_features)]
        rtd_prediction = self.concatenate_output(rtd_prediction)

        return rtd_prediction

    def get_encoder(self):
        return self.encoder


class TabTransformer(tf.keras.Model):
    def __init__(
        self,
        out_dim: int,
        out_activation: str,
        categorical_features: list = None,
        numerical_features: list = None,
        categorical_lookup: dict = None,
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        numerical_discretisers: dict = None,
        use_column_embedding: bool = True,
        mlp_hidden_factors: list = [2, 4],
        encoder = None
        
    ):

        super(TabTransformer, self).__init__()

        # Initialise encoder
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = TabTransformerEncoder(
                categorical_features,
                numerical_features,
                categorical_lookup,
                embedding_dim,
                depth,
                heads,
                attn_dropout,
                ff_dropout,
                numerical_discretisers,
                use_column_embedding,
            )

        # mlp layers
        if self.encoder.embed_numeric:
            mlp_input_dim = embedding_dim * (
                len(self.encoder.numerical) + len(self.encoder.categorical)
            )
        else:
            mlp_input_dim = len(self.encoder.numerical) + embedding_dim * len(self.encoder.categorical)

        self.mlp_final = build_mlp(mlp_input_dim, mlp_hidden_factors, ff_dropout)
        self.output_layer = Dense(out_dim, activation=out_activation)

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.mlp_final(x)
        output = self.output_layer(x)

        return output
