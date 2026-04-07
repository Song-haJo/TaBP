"""
Shared forward-pass-with-hooks utility for SSN and DDF.
"""

import torch


def forward_with_hooks(input_ids, model, mode: str, logits_processor=None):
    """
    Run a single forward pass, collecting each layer's hidden state via hooks.

    Args:
        input_ids:        Already-tokenized input tensor [1, seq_len].
        model:            HuggingFace causal LM.
        mode:             'latter' or 'whole'.
        logits_processor: Optional LogitsProcessor (SSN/ARC only).

    Returns:
        outputs:           GenerateOutput with .scores and .sequences.
        outputs_all_layers: {layer_idx: [hidden_state_tensor]}
    """
    start_layer = len(model.model.layers) // 2 if mode == "latter" else 0
    outputs_all_layers = {i: [] for i in range(start_layer, len(model.model.layers))}
    torch.cuda.empty_cache()

    def _hook(layer_num):
        def fn(module, inp, out):
            outputs_all_layers[layer_num].append(out[0].detach())
        return fn

    hooks = [
        layer.register_forward_hook(_hook(i))
        for i, layer in enumerate(model.model.layers[start_layer:], start=start_layer)
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            pad_token_id=model.generation_config.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            logits_processor=[logits_processor] if logits_processor is not None else [],
        )

    for h in hooks:
        h.remove()

    return outputs, outputs_all_layers
