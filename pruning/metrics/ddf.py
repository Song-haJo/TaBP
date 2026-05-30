"""
DDF (Directional Desirable Frequency) block ranking.

For each block b_i, DDF measures how often the block shifts the chosen
statistic in the *desirable* direction across forward passes:

    DDF(b_i) = (1/T) * Σ_t  1[ α · (s_i^(t) - s_{i-1}^(t)) > 0 ]

where T is the total number of forward passes, s_i^(t) is the metric value
probed at block i for pass t, and α is a sign coefficient that encodes the
direction of desirability per measure (see pruning.stats.MEASURE_DIRECTION).

Supported task types (set via load_calib_dataset):
    'qa'              – One forced forward pass per sample (ABCD answer only);
                        no sliding windows. Up to n_samples passes total.
    'text_generation' – Sliding windows + autoregressive generation;
                        n_windows × n_steps total passes.

Blocks with the *lowest* DDF scores are the least consistent at pushing
the statistic in the right direction and are therefore pruned first.
"""

from typing import Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..stats import compute_metric, MEASURE_DIRECTION
from ._hooks import forward_with_hooks


_WINDOW_SIZE = 32


def _score_blocks_qa(
    input_text, model, tokenizer, allowed_token_ids,
    logits_processor, key_token_id, mode: str, measure: str,
) -> list:
    """
    Compute per-block metric for one QA sample.

    Steps:
    1. Tokenize the prompt and run a forward pass with hooks,
       forcing generation to one of the answer-choice tokens.
    2. Treat the final block's logits as the target distribution.
    3. For every block:
       a. Feed the captured hidden state through the frozen lm_head.
       b. Apply softmax and restrict to the allowed answer tokens.
       c. Compute the chosen metric against the target distribution.
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

    # Step 2: final-layer logits as target (PATCHED: no .float() — keep fp16 like clean)
    final_logits = outputs.scores[-1]
    target_probs = F.softmax(final_logits, dim=-1).detach().cpu().numpy()[0]

    # Steps 3–4: probe each block with the model's own lm_head and compute metric
    block_scores = []
    for block_idx, block_outputs in block_outputs_all.items():
        if not block_outputs:
            continue

        block_output = block_outputs[-1]
        # PATCHED: do NOT unsqueeze; match clean's dim-based slicing exactly
        with torch.no_grad():
            lm_head_device = next(model.lm_head.parameters()).device
            block_logits = model.lm_head(block_output.to(lm_head_device))

        # PATCHED: replicate clean's branch on dim (2D → [0,:], 3D → [:,-1,:])
        # Also preserve RAW logits to pass into compute_metric (the cross_entropy
        # branch expects raw logits — passing softmaxed probs gives double softmax).
        if block_logits.dim() == 3:
            raw_logits_np = block_logits[:, -1, :].detach().cpu().numpy()[0]
            mask = torch.full_like(block_logits[:, -1, :], -float('inf')).to(block_logits.device)
            mask[:, allowed_token_ids] = 0
            masked_logits = block_logits[:, -1, :] + mask
            probs = F.softmax(masked_logits, dim=-1).detach().cpu().numpy()[0]
        elif block_logits.dim() == 2:
            raw_logits_np = block_logits[0, :].detach().cpu().numpy()
            mask = torch.full_like(block_logits[0, :], -float('inf')).to(block_logits.device)
            mask[allowed_token_ids] = 0
            masked_logits = block_logits[0, :] + mask
            probs = F.softmax(masked_logits, dim=-1).detach().cpu().numpy()
        else:
            raise ValueError(f"Unexpected block_logits dim: {block_logits.dim()}")
        block_probs = probs[allowed_token_ids]

        score = compute_metric(
            measure, block_probs, target_probs, raw_logits_np,  # PATCHED: raw logits, not probs
            allowed_token_ids, key_token_id
        )
        block_scores.append(score)
        print(f"  [DDF._score_qa] block {block_idx}: "
              f"probs(allowed)={[round(float(p),4) for p in block_probs]}  score={round(score,4)}")

    target_allowed = [round(float(target_probs[i]), 4) for i in allowed_token_ids]
    print(f"  [DDF._score_qa] target(allowed)={target_allowed}  "
          f"all_scores={[round(s,4) for s in block_scores]}")
    return block_scores


def _score_blocks_text_gen(
    input_ids, model, mode: str, measure: str,
) -> tuple:
    """
    Compute per-block metric for one autoregressive step (text generation mode).

    Steps:
    1. Run a one-step generate with hooks to capture each block's output.
    2. Treat the final generated token's distribution as the target.
    3. For every block:
       a. Feed the captured hidden state through the frozen lm_head.
       b. Apply softmax to get a full vocabulary distribution.
       c. Compute the chosen metric relative to the target distribution.
    4. Return (per-block scores, generation outputs).
    """
    outputs, block_outputs_all = forward_with_hooks(input_ids, model, mode)

    # Step 2: target distribution from the final generated token
    final_logits = outputs.scores[-1]
    target_probs = F.softmax(final_logits, dim=-1).float().detach().cpu().numpy()[0]

    # Steps 3–4: probe each block with the model's own lm_head
    block_scores = []
    for block_idx, block_outputs in block_outputs_all.items():
        if not block_outputs:
            continue
        block_output = block_outputs[-1]
        if block_output.dim() == 2:
            block_output = block_output.unsqueeze(0)

        with torch.no_grad():
            lm_head_device = next(model.lm_head.parameters()).device
            block_logits = model.lm_head(block_output.to(lm_head_device))

        raw_logits_np = block_logits[:, -1, :].float().detach().cpu().numpy()[0]  # PATCHED: keep raw
        probs = F.softmax(block_logits[:, -1, :], dim=-1).float().detach().cpu().numpy()[0]
        allowed_token_ids = list(range(len(probs)))

        score = compute_metric(
            measure, probs, target_probs, raw_logits_np,  # PATCHED: raw logits to logits arg
            allowed_token_ids, key_token_id=0,
        )
        block_scores.append(score)

    return block_scores, outputs


def rank_blocks(
    model,
    tokenizer,
    mode: str,
    measure: str,
    dataset,
    split: str,
    task_type: str,
    prepare_input: Optional[Callable] = None,
    prefix: str = "",
    postfix: str = "",
    n_samples: int = 1024,
    n_windows: int = 32,
    n_steps: int = 32,
    use_wandb: bool = False,
) -> np.ndarray:
    """
    Rank model blocks by DDF score.

    Supports two task types:

    QA ('qa'):
    1. Iterate over up to n_samples QA examples.
    2. For each sample, prepare_input formats it; run one forced forward pass.
    3. Count how often block i shifts the metric in the desirable direction
       relative to block i-1 (α · Δs > 0).
    4. Normalise counts by total samples → DDF ∈ [0, 1] per block.
    5. Sort blocks by ascending DDF: the least consistent blocks first.

    Text generation ('text_generation'):
    1. Concatenate the full corpus and split into sliding windows.
    2. For each window, autoregressively generate n_steps tokens.
    3–5. Same DDF counting, normalisation, and sorting as QA.

    Args:
        model:         HuggingFace causal LM.
        tokenizer:     Corresponding tokenizer.
        mode:          'latter' or 'whole'.
        measure:       Importance metric (see tabp.pruning.stats).
        dataset:       Pre-loaded HuggingFace dataset.
        split:         Dataset split name.
        task_type:     'qa' or 'text_generation'.
        prepare_input: Callable(example, tokenizer, prefix, postfix) that
                       returns the inputs for the appropriate scoring function.
                       Required for 'qa'.
        prefix:        Instruction prefix for QA prompts.
        postfix:       Text appended after answer choices.
        n_samples:     Max QA samples to use in 'qa' mode (default 1024).
        n_windows:     Number of sliding windows in 'text_generation' mode (default 32).
        n_steps:       Generation steps per window in 'text_generation' mode (default 32).

    Returns:
        Sorted numpy array of block indices (ascending DDF = most prunable first).
    """
    if measure not in MEASURE_DIRECTION:
        raise ValueError(
            f"Unknown measure '{measure}' for DDF. "
            f"Valid options: {list(MEASURE_DIRECTION)}"
        )
    alpha = MEASURE_DIRECTION[measure]

    model = model.eval()
    n_blocks = len(model.model.layers)
    start_block = n_blocks // 2 if mode == "latter" else 0

    print(f"[DDF] task_type={task_type}  mode={mode}  measure={measure}  "
          f"n_blocks={n_blocks}  start_block={start_block}  alpha={alpha}")

    block_ddf = np.zeros(n_blocks, dtype=float)
    total_steps = 0

    if task_type == "qa":
        # QA mode: one forward pass per sample, forced to answer tokens
        max_samples = min(n_samples, len(dataset[split]))
        for row in tqdm(range(max_samples), desc="Ranking blocks [DDF/QA]"):
            example = dataset[split][row]
            inputs, logits_processor, allowed_token_ids, key_token_id = \
                prepare_input(example, tokenizer, prefix, postfix)

            block_scores = _score_blocks_qa(
                inputs, model, tokenizer, allowed_token_ids,
                logits_processor, key_token_id, mode, measure,
            )
            torch.cuda.empty_cache()

            print(f"[DDF] sample {total_steps} block_scores ({len(block_scores)} blocks): "
                  f"{[round(s, 4) for s in block_scores]}")

            if use_wandb:
                import wandb
                log_dict = {f"ddf/block_{start_block + i}": block_scores[i]
                            for i in range(len(block_scores))}
                log_dict["ddf/sample"] = total_steps
                wandb.log(log_dict, step=total_steps)

            # Count directional shifts between adjacent blocks
            for i in range(1, len(block_scores)):
                delta = block_scores[i] - block_scores[i - 1]
                # PATCHED: count UNDESIRABLE shifts (penalty), matching clean's argsort(-penalty)
                if alpha * delta < 0:
                    block_ddf[start_block + i] += 1

            total_steps += 1

    else:  # text_generation
        # Sliding window + autoregressive generation
        first_param_device = next(model.parameters()).device
        full_text = "\n\n".join(dataset[split]["text"])
        token_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(first_param_device)

        prefix_ids = (
            tokenizer(prefix, return_tensors="pt").input_ids.to(first_param_device)
            if prefix else None
        )
        postfix_ids = (
            tokenizer(postfix, return_tensors="pt").input_ids.to(first_param_device)
            if postfix else None
        )

        for row in tqdm(range(n_windows), desc="Ranking blocks [DDF/TextGen]"):
            window = token_ids[:, row * _WINDOW_SIZE:(row + 1) * _WINDOW_SIZE]

            parts = [p for p in [prefix_ids, window, postfix_ids] if p is not None]
            inputs = torch.cat(parts, dim=1).long()

            for step in range(n_steps):
                block_scores, outputs = _score_blocks_text_gen(inputs, model, mode, measure)

                if total_steps == 0:
                    print(f"[DDF] window 0 step 0 block_scores ({len(block_scores)} blocks): "
                          f"{[round(s, 4) for s in block_scores]}")

                # Δs_i = score_i - score_{i-1} is attributed to block start_block + i
                for i in range(1, len(block_scores)):
                    delta = block_scores[i] - block_scores[i - 1]
                    if alpha * delta > 0:
                        block_ddf[start_block + i] += 1

                total_steps += 1

                # Append the newly generated token and continue
                generated_token = outputs.sequences[:, -1].unsqueeze(0)
                inputs = torch.cat([inputs, generated_token], dim=1)

    # Normalise to [0, 1]
    if total_steps > 0:
        block_ddf /= total_steps

    print("Block DDF scores:", block_ddf)

    if use_wandb:
        import wandb
        wandb.log({
            "ddf/block_scores": wandb.plot.bar(
                wandb.Table(
                    columns=["block", "ddf_score"],
                    data=[[i, float(block_ddf[i])] for i in range(n_blocks)],
                ),
                "block", "ddf_score", title="DDF: Block Scores",
            )
        })

    # PATCHED: match clean exactly — argsort(-penalty) over ALL blocks (descending)
    return np.argsort(-block_ddf)
