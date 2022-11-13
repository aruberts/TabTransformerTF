# TabTransformerTF

Implementation of [TabTransformer](https://arxiv.org/abs/2012.06678) in TensorFlow and Keras.

## Installation

The package can be install using
```python 
pip install tabtransformertf
```

## TabTransformer Usage

```python
from tabtransformertf.models.tabtransformer import TabTransformer
from tabtransformertf.utils.preprocessing import df_to_dataset, build_categorical_prep

# Category encoding layers
category_prep_layers = build_categorical_prep(train_data, CATEGORICAL_FEATURES)

# Preprocess dataset
train_dataset = df_to_dataset(train_data[FEATURES + [LABEL]], LABEL)

# Initialise model
tabtransformer = TabTransformer(
    numerical_features = NUMERIC_FEATURES,  # list with numerical features names
    categorical_features = CATEGORICAL_FEATURES,  # list with categorical features names
    categorical_lookup=category_prep_layers,  # dictionary with encoding layers
    embedding_dim=32,  
    out_dim=1, 
    out_activation='sigmoid',
    depth=4,
    heads=8,
    attn_dropout=0.2,
    ff_dropout=0.2,
    mlp_hidden_factors=[2, 4],
    use_column_embedding=True,  # flag to use fixed positional column embeddings
)

preds = tabtransformer.predict(train_dataset)
```
## FTTransformer Usage

```python
from tabtransformertf.models.fttransformer import FTTransformerEncoder, FTTransformer

# Encoder is specified separately in case we decide to pre-train the model
ft_linear_encoder = FTTransformerEncoder(
    numerical_features = NUMERIC_FEATURES,  # list of numeric features
    categorical_features = CATEGORICAL_FEATURES,  # list of numeric features
    categorical_lookup=category_prep_layers,  # dictionary of categorical lookup layers
    numerical_embeddings=None,  # None for linear embeddings
    numerical_embedding_type='linear',  # Numerical embedding type
    embedding_dim=16,  # Embedding dimension (for categorical, numerical, and contextual)
    depth=3,  # Number of Transformer Blocks (layers)
    heads=6,  # Number of attention heads in a Transofrmer Block
    attn_dropout=0.2,  # Dropout for attention layers
    ff_dropout=0.2,  # Dropout in Dense layers
    use_column_embedding=True,  # Fixed column embeddings
    explainable=True  # Whether we want to output attention importances or not
)

# Pass the encoder to the model
ft_linear_transformer = FTTransformer(
    encoder=ft_linear_encoder,  # Encoder from above
    out_dim=1,  # Number of outputs in final layer
    out_activation='sigmoid',  # Activation function for final layer
    final_layer_size=32,  # Pre-final layer, takes CLS contextual embeddings as input 
)

preds = ft_linear_transformer.predict(train_dataset)
```


## Credits

As a reference, I've combined [this implementation](https://github.com/CahidArda/tab-transformer-keras) with [Keras guide](https://keras.io/examples/structured_data/tabtransformer/).
