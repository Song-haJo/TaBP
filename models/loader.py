"""
Model loading utilities for TaBP (FluctARCE).

Accepts any HuggingFace model ID or local path — no hardcoded shortcuts.

Examples:
    load_model("mistralai/Mistral-7B-v0.3")
    load_model("meta-llama/Meta-Llama-3-8B")
    load_model("google/gemma-2-9b")
    load_model("microsoft/phi-2")
    load_model("Qwen/Qwen2.5-7B")
    load_model("/path/to/local/model")
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor


class ForcedTokenLogitsProcessor(LogitsProcessor):
    """Restrict next-token generation to a fixed set of token IDs."""

    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, -float("inf"))
        mask[:, self.allowed_token_ids] = 0
        return scores + mask


def load_model(model_path: str, use_cache: bool = False, torch_dtype: str = "auto"):
    """
    Load a causal LM and its tokenizer from HuggingFace Hub or a local path.

    Args:
        model_path:  HuggingFace repo ID (e.g. 'mistralai/Mistral-7B-v0.3')
                     or an absolute path to a local model directory.
        use_cache:   Enable KV cache (set False for layer-ranking; True after pruning).
        torch_dtype: Torch dtype string passed to from_pretrained.
                     'auto' lets the library decide; use 'float16' to save VRAM.

    Returns:
        (model, tokenizer)
    """
    dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=dtype,
        use_cache=use_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
