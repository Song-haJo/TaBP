# pruning

Block importance measurement, ranking, and removal logic.

---

## stats.py

Functions originally defined at the top level of `PRUNE.py`. In the original, the `if measure == "..."` branches were inlined inside `compute_layer_metrics`.

| Function | Original name | Changes |
|---|---|---|
| `compute_entropy` | `compute_entropy` | unchanged |
| `compute_confidence` | `compute_confidence` | unchanged |
| `compute_gap` | `compute_gap` | unchanged |
| `compute_answerkey_prob` | `compute_answerkey_prob` | unchanged |
| `compute_cross_entropy` | `compute_cross_entropy` | removed hardcoded `"cuda:3"` → uses `torch.cuda.is_available()` |
| `compute_kl_divergence` | `compute_kl_divergence` | unchanged |
| `compute_js_divergence` | `compute_js_divergence` | unchanged |
| `compute_earth_movers_distance` | `compute_earth_movers_distance` | unchanged |
| `compute_total_variation_distance` | `compute_total_variation_distance` | unchanged |
| `compute_hellinger_distance` | `compute_hellinger_distance` | unchanged |
| `compute_cosine_similarity` | `compute_cosine_similarity` | unchanged |
| `compute_bhattacharyya_distance` | `compute_bhattacharyya_distance` | unchanged |
| `compute_energy_distance` | `compute_energy_distance` | unchanged |
| `compute_metric` | (if-elif branches inside `compute_layer_metrics` in `PRUNE.py`) | extracted into a standalone dispatcher function |
| `MEASURE_ABBR` | — | new. abbreviation map for argparse choices and result directory naming |

---

## metrics/

Statistical Shift Norm (SSN) and Directional Desirable Frequency (DDF) block ranking implementations. Both use `forward_with_hooks` from `_hooks.py`.

See [metrics/README.md](metrics/README.md) for per-function details.

### _hooks.py

| Function | Original location | Original name | Changes |
|---|---|---|---|
| `forward_with_hooks` | `PRUNE.py`, `PRUNE_penalty.py` | `generate_with_hooks` | Merged hook logic defined separately in each file. Accepts `input_ids` directly so it works for both SSN and DDF. `logits_processor` made optional. |

### ssn.py

Source: `rank_layers` and `compute_layer_metrics` from `PRUNE.py`.

| Function | Original name | Changes |
|---|---|---|
| `_score_blocks_qa` | `compute_layer_metrics` | Renamed layer→block. Accepts `lm_heads` dict with `frozen`/`trained` branching. Removed hardcoded device mapping. |
| `_score_blocks_text_gen` | — | New. Adds text generation support using entropy over top-k vocabulary. |
| `rank_blocks` | `rank_layers` | Renamed layer→block. Added `task_type` and `prepare_input` parameters for dataset-agnostic dispatch. |

### ddf.py

Source: `calculate_layer_penalties` from `PRUNE_penalty.py`.

| Function | Original name | Changes |
|---|---|---|
| `_score_blocks_qa` | — | New. Adds QA support (was WikiText-only in the original). |
| `_score_blocks_text_gen` | `compute_layer_metrics` + inlined `generate_with_hooks` | Renamed layer→block. Replaced with `forward_with_hooks` call. |
| `rank_blocks` | `calculate_layer_penalties` | Renamed layer→block. Added `task_type` and `prepare_input`. QA mode uses `n_samples`; text generation mode uses `n_windows` × `n_steps`. |

---

## pruner.py

Source: `prune_layers` from `PRUNE.py`.

| Function | Original name | Changes |
|---|---|---|
| `prune_layers` | `prune_layers` | Added type check to handle both `np.ndarray` and `list`. Added completion message. |
