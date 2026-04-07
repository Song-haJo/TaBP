# data

Dataset loading and input formatting utilities.

---

## preprocessing.py

### Dataset Registry

`_DATASET_REGISTRY` maps a dataset key to everything needed to use it:

```python
{
    "arc_easy": {
        "hf_name": "ai2_arc",
        "hf_config": "ARC-Easy",
        "split": "train",
        "task_type": "qa",
        "prepare_input": prepare_arc_inputs,
    },
    "wikitext": {
        "hf_name": "wikitext",
        "hf_config": "wikitext-2-raw-v1",
        "split": "train",
        "task_type": "text_generation",
        "prepare_input": _prepare_wikitext,
    },
}
```

`task_type` controls which scoring path is used inside `rank_blocks`:
- `"qa"` — one forward pass per sample with forced answer tokens (no sliding windows)
- `"text_generation"` — sliding window + autoregressive generation

To add a new dataset, write a `prepare_input` function and add an entry to the registry.

---

### Functions

| Function | Description |
|---|---|
| `load_calib_dataset(calib_data, dataset_path=None)` | Returns `(dataset, split, task_type, prepare_input)`. Loads from HuggingFace Hub or a local path via `load_from_disk`. |
| `prepare_arc_inputs(example, tokenizer, prefix, postfix)` | Formats one AI2 ARC-Easy sample. Returns `(input_str, logits_processor, allowed_token_ids, key_token_id)`. |

`prepare_input` callables all share the same signature:
```python
prepare_input(example, tokenizer, prefix, postfix)
```
Each function is responsible for all dataset-specific formatting (number of choices, presence of context, field names, etc.).

---

### History

| Function | Original location | Original name | Changes |
|---|---|---|---|
| `load_calib_dataset` | inline inside `rank_layers` in `PRUNE.py` and `calculate_layer_penalties` in `PRUNE_penalty.py` | (inline `load_from_disk` / `load_dataset`) | Originally each function loaded its dataset internally with a hardcoded path. Extracted and generalized: `calib_data` selects a registered dataset; now returns `(dataset, split, task_type, prepare_input)` |
| `prepare_arc_inputs` | `PRUNE.py` → inline `prepare_inputs`, `EVAL.py` → `prepare_inputs` | `prepare_inputs` | Merged duplicate definitions. Moved here from `models/loader.py` (input formatting belongs in data preprocessing, not model loading) |
