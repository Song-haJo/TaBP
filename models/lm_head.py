"""
Per-layer lm_head utilities for TaBP (FluctARCE).

Two modes for obtaining an lm_head at a given intermediate layer:

    'frozen'  – Use the model's own lm_head as-is (no extra files needed).
    'trained' – Load a per-layer trained lm_head checkpoint from disk.
                Checkpoints must follow the naming convention:
                    <ckpt_dir>/lm_head_prune<layer_num>.pt
"""

import os

import torch
import torch.nn as nn


def create_lm_head(model) -> nn.Linear:
    """Instantiate a fresh lm_head matching the model's hidden and vocab size."""
    return nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False)


def load_trained_lm_head(layer_num: int, model, device, ckpt_dir: str) -> nn.Linear:
    """
    Load a pre-trained per-layer lm_head checkpoint.

    Args:
        layer_num: Layer index the checkpoint was trained for.
        model:     Base model (used to infer hidden_size / vocab_size).
        device:    Target device for the loaded weights.
        ckpt_dir:  Directory containing the .pt checkpoint files.

    Returns:
        Loaded and eval-mode nn.Linear on `device`.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    ckpt_path = os.path.join(ckpt_dir, f"lm_head_prune{layer_num}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"lm_head checkpoint not found: {ckpt_path}\n"
            "Train per-layer lm_heads first, or use --lm_head_type frozen."
        )
    lm_head = create_lm_head(model)
    lm_head.load_state_dict(torch.load(ckpt_path, map_location=device))
    return lm_head.to(device).eval()


def build_lm_heads(model, start_layer: int, lm_head_type: str,
                   lm_head_ckpt_dir: str = None) -> dict:
    """
    Build a {layer_num: lm_head} dict for all layers from start_layer onward.

    Args:
        model:           Base causal LM.
        start_layer:     First layer index to include.
        lm_head_type:    'frozen' or 'trained'.
        lm_head_ckpt_dir: Required when lm_head_type='trained'.

    Returns:
        Dict mapping layer index → nn.Linear, or None when lm_head_type='frozen'.
    """
    if lm_head_type == "frozen":
        return None   # caller uses model.lm_head directly

    if lm_head_ckpt_dir is None:
        raise ValueError(
            "lm_head_ckpt_dir must be provided when lm_head_type='trained'."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_layers = len(model.model.layers)
    return {
        i: load_trained_lm_head(i, model, device, lm_head_ckpt_dir)
        for i in range(start_layer, n_layers)
    }
