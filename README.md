# TaBP: Task-aware Block Pruning

EACL 2026 Findings #1237

A structured pruning method that measures the fluctuation of each block's output distribution using calibration data and removes blocks with low importance.

Two ranking strategies:
- **SSN** (Statistical Shift Norm): ranks blocks by the average magnitude of distributional shift, regardless of direction
- **DDF** (Directional Desirable Frequency): ranks blocks by how frequently they produce desirable directional shifts in the output distribution

---

## Project Structure

```
TaBP/
├── main.py                     # Entry point
├── requirements.txt
├── run.sh                      # Batch experiment script
├── lm_eval/                    # Evaluation framework
│   ├── eval_ppl.py
│   └── eval_zeroshot_acc.py
├── models/                     # Model loading, lm_head utilities
├── data/                       # Dataset loading, input formatting
└── pruning/                    # Ranking and pruning logic
    ├── stats.py                # Distribution-based metric functions
    ├── pruner.py               # Block removal
    └── metrics/                # SSN / DDF ranking implementations
        ├── _hooks.py           # Shared forward-with-hooks utility
        ├── ssn.py              # SSN ranking (rank_blocks)
        └── ddf.py              # DDF ranking (rank_blocks)
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# SSN
python main.py \
    --model_name mistralai/Mistral-7B-v0.3 \
    --method ssn \
    --mode latter \
    --measure entropy

# DDF
python main.py \
    --model_name mistralai/Mistral-7B-v0.3 \
    --method ddf \
    --mode latter \
    --measure entropy
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--model_name` | HuggingFace model ID or local path | required |
| `--method` | `ssn` or `ddf` | required |
| `--mode` | `latter` (second-half blocks only) or `whole` (all blocks) | required |
| `--measure` | Block importance metric (see below) | required |
| `--calib_data` | Calibration dataset for both SSN and DDF: `arc_easy` (QA mode) or `wikitext` (text generation mode). Register additional datasets in `data/preprocessing.py`. | `arc_easy` |
| `--dataset_path` | Local dataset path (omit to load from HuggingFace Hub) | None |
| `--n_samples` | Number of calibration samples (SSN; DDF in QA mode) | 1024 |
| `--n_windows` | Number of sliding windows (DDF text generation mode only) | 32 |
| `--n_steps` | Generation steps per window (DDF text generation mode only) | 32 |
| `--prune_counts` | List of block counts to prune and evaluate | `1 3 5 6 7` |
| `--lm_head_type` | SSN only. `frozen` or `trained` | `frozen` |
| `--lm_head_ckpt_dir` | SSN only. Directory of trained lm_head checkpoints | None |

### Measure Options

`entropy` | `confidence_score` | `gap` | `key` | `cross_entropy` | `kl_divergence` | `js_divergence` | `wasserstein_dist` | `hellinger_dist` | `bhattacharyya_dist` | `cosine_similarity` | `tvd` | `energy`

---

## Adding a New Calibration Dataset

1. Write a `prepare_input(example, tokenizer, prefix, postfix)` function in `data/preprocessing.py`.
   - **QA datasets**: return `(input_str, logits_processor, allowed_token_ids, key_token_id)`
   - **Text generation datasets**: return the text string
2. Register it in `_DATASET_REGISTRY` with the correct `task_type` (`"qa"` or `"text_generation"`).

That's all. `--calib_data` will automatically include the new key as a valid choice.
