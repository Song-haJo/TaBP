# pruning/metrics

Statistical Shift Norm (SSN) and Directional Desirable Frequency (DDF) block ranking implementations. Both share the common hook utility in `_hooks.py`.

---

## _hooks.py

| Function | Original location | Original name | Changes |
|---|---|---|---|
| `forward_with_hooks` | `PRUNE.py`, `PRUNE_penalty.py` | `generate_with_hooks` | Merged hook logic defined separately in each file. Accepts `input_ids` directly so it works for both SSN and DDF. `logits_processor` made optional. |

---

## ssn.py

Source: `rank_layers` and `compute_layer_metrics` from `PRUNE.py`.

### Functions

| Function | Description |
|---|---|
| `_score_blocks_qa(input_text, model, tokenizer, allowed_token_ids, logits_processor, key_token_id, mode, measure, lm_heads=None)` | One forward pass for a QA sample. Returns per-block metric scores computed over the allowed answer tokens. |
| `_score_blocks_text_gen(input_text, model, tokenizer, mode, lm_heads=None, top_k=50)` | One forward pass for a text generation sample. Returns per-block top-k entropy. |
| `rank_blocks(...)` | Ranks blocks by SSN fluctuation score. Dispatches to `_score_blocks_qa` or `_score_blocks_text_gen` based on `task_type`. |

### History

| Function | Original name | Changes |
|---|---|---|
| `_score_blocks_qa` | `compute_layer_metrics` | Renamed layer→block. Accepts `lm_heads` dict with `frozen`/`trained` branching. Removed hardcoded device mapping (`get_target_device`). |
| `_score_blocks_text_gen` | — | New. Adds text generation support using entropy over top-k vocabulary. |
| `rank_blocks` | `rank_layers` | Renamed layer→block. Removed hardcoded paths and sample counts. Added `task_type` and `prepare_input` parameters for dataset-agnostic dispatch. `prepare_input` is a callable registered per dataset in `data/preprocessing.py`. |

---

## ddf.py

Source: `calculate_layer_penalties` from `PRUNE_penalty.py`.

### Functions

| Function | Description |
|---|---|
| `_score_blocks_qa(input_text, model, tokenizer, allowed_token_ids, logits_processor, key_token_id, mode, measure)` | One forward pass for a QA sample. Returns per-block metric scores over the allowed answer tokens. |
| `_score_blocks_text_gen(input_ids, model, mode, measure)` | One autoregressive generation step. Returns `(per-block scores, outputs)` over the full vocabulary. |
| `rank_blocks(...)` | Ranks blocks by DDF score. QA mode: one pass per sample, no sliding windows. Text generation mode: sliding window + autoregressive generation. |

### History

| Function | Original name | Changes |
|---|---|---|
| `_score_blocks_qa` | — | New. Adds QA support (was WikiText-only in the original). |
| `_score_blocks_text_gen` | `compute_layer_metrics` + inlined `generate_with_hooks` | Renamed layer→block. Replaced with `forward_with_hooks` call. |
| `rank_blocks` | `calculate_layer_penalties` | Renamed layer→block. Added `task_type` and `prepare_input` parameters. QA mode uses `n_samples`; text generation mode uses `n_windows` × `n_steps`. Duplicate metric functions removed → uses `compute_metric` from `stats.py`. |
