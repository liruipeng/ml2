# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %%
import torch.nn as nn

def get_activation(name: str):
    name = name.lower()
    activations = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'silu': nn.SiLU,
        'gelu': nn.GELU,
        'softmax': lambda: nn.Softmax(dim=1),  # safer default
    }
    if name not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    return activations[name]()
