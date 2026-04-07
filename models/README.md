# models

Model loading and lm_head utilities.

---

## loader.py

| Function / Class | Original location | Original name | Changes |
|---|---|---|---|
| `ForcedTokenLogitsProcessor` | `PRUNE.py`, `EVAL.py` | `ForcedTokenLogitsProcessor` | Unchanged. Merged duplicate definitions from both files. |
| `load_model` | `EVAL.py` | `load_model` | Original accepted abbreviations (`"M"`, `"L"`, `"G"`) mapped to hardcoded local paths. Generalized to use `AutoModelForCausalLM` accepting any HuggingFace model ID or local path directly. |

---

## lm_head.py

Used by SSN only (`rank_blocks` in `ssn.py`). DDF always uses the model's own `lm_head` directly.

| Function | Original location | Original name | Changes |
|---|---|---|---|
| `create_lm_head` | `PRUNE.py` | `create_lm_head` | Unchanged. |
| `load_trained_lm_head` | `PRUNE.py` | `load_lm_head_for_layer` | Removed hardcoded path (`/home/jos02/ft/lm_head_checkpoints_M`). Generalized with `ckpt_dir` argument. Added `FileNotFoundError`. |
| `build_lm_heads` | `PRUNE.py` → inline inside `rank_layers` | (inline) | Extracted the inline `lm_heads = {}` construction from `rank_layers` into a standalone function. Returns `None` in `frozen` mode (caller uses `model.lm_head` directly). |
