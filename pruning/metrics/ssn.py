"""
SSN (Statistical Shift Norm) block ranking.

For each block b_i, SSN measures how much the chosen statistic
shifts as a result of that block across calibration samples:

    SSN(b_i) = (1/N) * Σ_n |s_i^(n) - s_{i-1}^(n)|

where s_i^(n) is the metric value probed at block i on sample n.
Blocks with the *smallest* SSN scores are the most redundant and
are therefore pruned first.

Supported task types (set via load_calib_dataset):
    'qa'              – One forced forward pass per sample; scores are
                        computed over the allowed answer tokens.
    'text_generation' – One forward pass per text sample; scores are
                        measured as top-k entropy over the full vocabulary.

lm_head options (lm_head_type):
    'frozen'  – Use the model's own lm_head as-is (no checkpoints needed).
    'trained' – Load per-block trained lm_head checkpoints from disk
                (requires lm_head_ckpt_dir).

Calibration data:
    Pass a pre-loaded HuggingFace dataset, split name, task_type, and
    prepare_input callable from load_calib_dataset().
"""

import gc
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..stats import compute_metric, compute_entropy
from models.lm_head import build_lm_heads
from ._hooks import forward_with_hooks


def _score_blocks_qa(
    input_text, model, tokenizer, allowed_token_ids,
    logits_processor, key_token_id, mode: str, measure: str,
    lm_heads: Optional[dict] = None,
    use_wandb: bool = False,
    sample_idx: int = 0,
) -> list:
    """
    Compute the chosen metric at each block for one QA sample.

    Steps:
    1. Tokenize the prompt and run a forward pass, capturing each
       block's hidden-state output via registered hooks.
       Generation is forced to one of the allowed answer tokens.
    2. Treat the final block's logits as the ground-truth distribution.
    3. For every block:
       a. Feed the captured hidden state through lm_head to get logits.
       b. Apply softmax and restrict to the allowed answer tokens.
       c. Compute the chosen metric against the ground-truth distribution.
    4. Return one scalar per block.
    """
    first_param_device = next(model.parameters()).device
    input_ids = tokenizer(
        input_text, return_tensors="pt", padding=True, truncation=True
    ).to(first_param_device)["input_ids"]

    # Step 1: forward pass with hooks to capture all block outputs
    outputs, block_outputs_all = forward_with_hooks(
        input_ids, model, mode, logits_processor
    )

    # Step 2: final-layer logits as target
    final_logits = outputs.scores[-1]
    target_probs = F.softmax(final_logits, dim=-1).float().detach().cpu().numpy()[0]

    # Steps 3–4: probe each block with lm_head and compute metric
    block_scores = []
    block_probs_list = []  # store for wandb logging
    for block_idx, block_outputs in block_outputs_all.items():
        if not block_outputs:
            continue

        block_output = block_outputs[-1]
        if block_output.dim() == 2:
            block_output = block_output.unsqueeze(0)

        with torch.no_grad():
            if lm_heads is not None:
                head = lm_heads[block_idx]
                device = next(head.parameters()).device
                block_logits = head(block_output.to(device))
            else:
                lm_head_device = next(model.lm_head.parameters()).device
                block_logits = model.lm_head(block_output.to(lm_head_device))

        logits = F.softmax(block_logits[:, -1, :], dim=-1).float().detach().cpu().numpy()[0]
        block_probs_raw = logits[allowed_token_ids]
        block_probs = block_probs_raw / (block_probs_raw.sum() + 1e-10)

        score = compute_metric(
            measure, block_probs, target_probs, logits,
            allowed_token_ids, key_token_id
        )
        block_scores.append(score)
        block_probs_list.append((block_idx, block_probs))
        print(f"  [SSN._score_qa] block {block_idx}: "
              f"probs(allowed)={[round(float(p),4) for p in block_probs]}  score={round(score,4)}")

    target_allowed = [round(float(target_probs[i]), 4) for i in allowed_token_ids]
    print(f"  [SSN._score_qa] target(allowed)={target_allowed}  "
          f"all_scores={[round(s,4) for s in block_scores]}")

    if use_wandb:
        import wandb
        n_opts = len(allowed_token_ids)
        opt_cols = [f"opt_{i}_prob" for i in range(n_opts)]
        columns = ["block"] + opt_cols + ["metric"]
        rows = []
        for (block_idx, block_probs), score in zip(block_probs_list, block_scores):
            rows.append([block_idx] + block_probs.tolist() + [score])
        target_norm = np.array([target_probs[i] for i in allowed_token_ids])
        target_norm = target_norm / (target_norm.sum() + 1e-10)
        rows.append(["target"] + target_norm.tolist() + [float("nan")])
        wandb.log({
            f"ssn_detail/sample_{sample_idx}": wandb.Table(columns=columns, data=rows)
        }, step=sample_idx)

    return block_scores


def _score_blocks_text_gen(
    input_text, model, tokenizer, mode: str,
    lm_heads: Optional[dict] = None,
    top_k: int = 50,
) -> list:
    """
    Compute top-k entropy at each block for one text generation sample.

    Steps:
    1. Tokenize (truncated) and run a forward pass with hooks.
    2. For every block:
       a. Feed the hidden state through lm_head to get vocabulary logits.
       b. Take the top-k probability mass and renormalise.
       c. Measure entropy of that focused distribution.
    3. Return one entropy value per block.
    """
    first_param_device = next(model.parameters()).device
    input_ids = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=64
    ).to(first_param_device)["input_ids"]

    # Step 1: forward pass with hooks
    _, block_outputs_all = forward_with_hooks(input_ids, model, mode)

    # Steps 2–3: probe each block and compute entropy
    block_scores = []
    for block_idx, block_outputs in block_outputs_all.items():
        if not block_outputs:
            continue
        block_output = block_outputs[-1]
        if block_output.dim() == 2:
            block_output = block_output.unsqueeze(0)

        with torch.no_grad():
            if lm_heads is not None:
                head = lm_heads[block_idx]
                device = next(head.parameters()).device
                block_logits = head(block_output.to(device))
            else:
                lm_head_device = next(model.lm_head.parameters()).device
                block_logits = model.lm_head(block_output.to(lm_head_device))

        probs = F.softmax(block_logits[:, -1, :], dim=-1).float().detach().cpu().numpy()[0]
        top_ids = np.argsort(probs)[-top_k:]
        top_probs = probs[top_ids] / probs[top_ids].sum()
        block_scores.append(compute_entropy(top_probs))

    return block_scores


def rank_blocks(
    model,
    tokenizer,
    prefix: str,
    postfix: str,
    mode: str,
    measure: str,
    dataset,
    split: str,
    task_type: str,
    prepare_input: Optional[Callable] = None,
    n_samples: int = 1024,
    lm_head_type: str = "frozen",
    lm_head_ckpt_dir: Optional[str] = None,
    use_wandb: bool = False,
) -> np.ndarray:
    """
    Rank model blocks by SSN fluctuation score.

    Steps:
    1. Build optional per-block lm_head probes (frozen or trained checkpoints).
    2. For each calibration sample, run a forward pass and collect per-block
       metric scores.
       - 'qa':              prepare_input formats the sample; scores are computed
                            over forced answer tokens.
       - 'text_generation': scores are measured as top-k entropy.
    3. For adjacent blocks (b_{i-1}, b_i), record the absolute difference
       |score_i - score_{i-1}|. This is the fluctuation attributed to block i.
    4. Average fluctuations over all N samples → one importance score per block.
    5. Sort blocks by ascending score: the least fluctuating (most redundant)
       blocks are returned first as pruning candidates.

    Args:
        model:            HuggingFace causal LM.
        tokenizer:        Corresponding tokenizer.
        prefix:           Instruction prefix for QA prompts.
        postfix:          Suffix after answer choices (e.g., 'Answer: ').
        mode:             'latter' (prune second half) or 'whole' (any block).
        measure:          Importance metric (see tabp.pruning.stats).
        dataset:          Pre-loaded HuggingFace dataset.
        split:            Dataset split name (e.g., 'train').
        task_type:        'qa' or 'text_generation'.
        prepare_input:    Callable(example, tokenizer, prefix, postfix) that
                          returns the inputs for _score_blocks_qa or the text
                          string for _score_blocks_text_gen. Required for 'qa'.
        n_samples:        Number of calibration samples (default 1024).
        lm_head_type:     'frozen' or 'trained'.
        lm_head_ckpt_dir: Checkpoint dir; required when lm_head_type='trained'.

    Returns:
        Sorted numpy array of block indices (ascending SSN = most prunable first).
    """
    model = model.eval()
    n_blocks = len(model.model.layers)
    start_block = n_blocks // 2 if mode == "latter" else 0

    print(f"[SSN] task_type={task_type}  mode={mode}  measure={measure}  "
          f"n_blocks={n_blocks}  start_block={start_block}  n_samples={n_samples}")

    # Step 1: build lm_head probes for intermediate blocks
    lm_heads = build_lm_heads(model, start_block, lm_head_type, lm_head_ckpt_dir)

    # Step 2–3: accumulate per-block fluctuations over calibration samples
    block_fluctuations = np.zeros((n_blocks, n_samples))
    valid_rows = 0

    for row in tqdm(range(len(dataset[split])), desc="Ranking blocks [SSN]"):
        if valid_rows >= n_samples:
            break

        example = dataset[split][row]

        if task_type == "qa":
            # prepare_input handles all dataset-specific formatting
            inputs, logits_processor, allowed_token_ids, key_token_id = \
                prepare_input(example, tokenizer, prefix, postfix)
            block_scores = _score_blocks_qa(
                inputs, model, tokenizer, allowed_token_ids,
                logits_processor, key_token_id, mode, measure, lm_heads,
                use_wandb=use_wandb, sample_idx=valid_rows,
            )
        else:  # text_generation
            text = prepare_input(example, tokenizer, prefix, postfix)
            if not text:
                continue
            block_scores = _score_blocks_text_gen(
                text, model, tokenizer, mode, lm_heads
            )

        print(f"[SSN] sample {valid_rows} block_scores ({len(block_scores)} blocks): "
              f"{[round(s, 4) for s in block_scores]}")

        if use_wandb:
            import wandb
            log_dict = {f"ssn/block_{start_block + i}": block_scores[i]
                        for i in range(len(block_scores))}
            log_dict["ssn/sample"] = valid_rows
            wandb.log(log_dict, step=valid_rows)

        torch.cuda.empty_cache()

        # |score_i - score_{i-1}| is credited to block start_block + i
        diffs = [abs(block_scores[i] - block_scores[i - 1])
                 for i in range(1, len(block_scores))]
        for i, diff in enumerate(diffs):
            block_fluctuations[start_block + i + 1, valid_rows] = diff
        valid_rows += 1

    if lm_heads is not None:
        del lm_heads
    gc.collect()
    torch.cuda.empty_cache()

    # Step 4: average over samples
    avg_fluctuations = np.mean(block_fluctuations, axis=1)
    print("Average block fluctuations:", avg_fluctuations)

    if use_wandb:
        import wandb
        wandb.log({
            "ssn/avg_fluctuation": wandb.plot.bar(
                wandb.Table(
                    columns=["block", "avg_fluctuation"],
                    data=[[i, float(avg_fluctuations[i])] for i in range(n_blocks)],
                ),
                "block", "avg_fluctuation", title="SSN: Avg Fluctuation per Block",
            )
        })

    # Step 5: sort by ascending fluctuation (most prunable first)
    if mode == "latter":
        mid_point = n_blocks // 2 + 1
        fixed_blocks = list(range(mid_point))
        sorted_latter = np.argsort(avg_fluctuations)
        sorted_latter = [b for b in sorted_latter if b not in fixed_blocks]
        sorted_blocks = np.concatenate((fixed_blocks, sorted_latter))
        return sorted_blocks[len(sorted_blocks) // 2 + 1:]
    else:
        return np.argsort(avg_fluctuations)[1:]
