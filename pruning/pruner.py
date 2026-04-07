"""
Layer pruning for FluctARCE.

Physically removes transformer layers from a causal LM and updates
the model's internal bookkeeping (attention indices, config).
"""

import numpy as np


def prune_layers(model, layers_to_prune) -> None:
    """
    Remove the specified layers from the model in-place.

    Args:
        model:          A HuggingFace causal LM with model.model.layers.
        layers_to_prune: Array or list of layer indices to remove.
    """
    indices = (
        layers_to_prune.tolist()
        if isinstance(layers_to_prune, np.ndarray)
        else list(layers_to_prune)
    )
    if not indices:
        return

    for layer_idx in sorted(indices, reverse=True):
        del model.model.layers[layer_idx]
        for layer in model.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and getattr(attn, "layer_idx", None) is not None:
                if attn.layer_idx > layer_idx:
                    attn.layer_idx -= 1

    model.config.num_hidden_layers = len(model.model.layers)
    print(f"Pruned {len(indices)} layer(s). Remaining: {len(model.model.layers)}")
