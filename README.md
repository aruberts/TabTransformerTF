# TabTransformerTF

Implementation of [TabTransformer](https://arxiv.org/abs/2012.06678) in TensorFlow and Keras.

## Installation

The package can be install using
```python 
pip install tabtransformertf
```

## Usage

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
    numerical_discretisers=None,  # simply passing the numeric features like in original paper
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

## Credits

As a reference, I've combined [this implementation](https://github.com/CahidArda/tab-transformer-keras) with [Keras guide](https://keras.io/examples/structured_data/tabtransformer/).
