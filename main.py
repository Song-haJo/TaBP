"""
TaBP (Task-aware Block Pruning) — Main Pruning & Evaluation Pipeline
=====================================================================

Usage:
    # SSN (Soft Score Narrowing):
    python main.py \\
        --model_name mistralai/Mistral-7B-v0.3 \\
        --method ssn \\
        --mode latter \\
        --measure entropy

    # DDF (Difference-based Distribution Fluctuation):
    python main.py \\
        --model_name mistralai/Mistral-7B-v0.3 \\
        --method ddf \\
        --mode latter \\
        --measure entropy

Arguments:
    --model_name      HuggingFace model ID or local path.
                      Examples: mistralai/Mistral-7B-v0.3
                                meta-llama/Meta-Llama-3-8B
                                google/gemma-2-9b
                                microsoft/phi-2
                                Qwen/Qwen2.5-7B
    --method          Ranking method:
                      'ssn' – Soft Score Narrowing (fluctuation average)
                      'ddf' – Difference-based Distribution Fluctuation (penalty count)
    --mode            'latter' (prune second half of layers) or
                      'whole'  (any layer)
    --measure         Layer importance metric. Choices:
                      entropy | confidence_score | gap | key |
                      cross_entropy | kl_divergence | js_divergence |
                      wasserstein_dist | hellinger_dist | bhattacharyya_dist |
                      cosine_similarity | tvd | energy
    --calib_data      Calibration dataset for block ranking:
                      'arc_easy'  – AI2 ARC-Easy, QA mode (default)
                      'wikitext'  – WikiText-2, text generation mode
                      Add new datasets in data/preprocessing.py.
    --dataset_path    (optional) Local path to the calibration dataset
                      (load_from_disk). Omit to load from HuggingFace Hub.
    --n_samples       Number of calibration samples for SSN (default: 1024).
    --n_windows       Number of sliding windows for DDF (default: 32).
    --n_steps         Generation steps per window for DDF (default: 32).
    --prune_counts    Space-separated pruning counts to evaluate
                      (default: 1 3 5 6 7).
    --lm_head_type    SSN only. 'frozen' – use the model's own lm_head as-is (default).
                      'trained' – load per-layer trained lm_head checkpoints
                                  (requires --lm_head_ckpt_dir).
"""

import argparse
import gc
import json
import os

import torch

from models import load_model
from pruning import prune_layers, rank_blocks_ssn, rank_blocks_ddf
from pruning.stats import MEASURE_ABBR
from data import load_calib_dataset, _DATASET_REGISTRY
from lm_eval.eval_ppl import eval_ppl, generate_txt
from lm_eval.eval_zeroshot_acc import eval_acc


TASK_LIST = [
    "arc_easy",
    "arc_challenge",
    "boolq",
    "hellaswag",
    "piqa",
    "copa",
    "winogrande",
]

PREFIX = "Answer the following multiple choice question by giving the most appropriate response.\n"
POSTFIX = "Answer: "


def parse_args():
    parser = argparse.ArgumentParser(description="TaBP: Layer Pruning Pipeline")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model ID or local path "
                             "(e.g. mistralai/Mistral-7B-v0.3)")
    parser.add_argument("--method", type=str, required=True,
                        choices=["ssn", "ddf"],
                        help="Ranking method: 'ssn' or 'ddf'")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["latter", "whole"],
                        help="Pruning scope: 'latter' or 'whole'")
    parser.add_argument("--measure", type=str, required=True,
                        choices=list(MEASURE_ABBR.keys()),
                        help="Layer importance metric")
    parser.add_argument("--calib_data", type=str, default="arc_easy",
                        choices=list(_DATASET_REGISTRY),
                        help="Calibration dataset (default: arc_easy). "
                             "Register new datasets in data/preprocessing.py.")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Local path to calibration dataset (load_from_disk). "
                             "If omitted, loads from HuggingFace Hub.")
    parser.add_argument("--n_samples", type=int, default=1024,
                        help="SSN: number of calibration samples (default: 1024)")
    parser.add_argument("--n_windows", type=int, default=32,
                        help="DDF: number of sliding windows (default: 32)")
    parser.add_argument("--n_steps", type=int, default=32,
                        help="DDF: generation steps per window (default: 32)")
    parser.add_argument("--prune_counts", type=int, nargs="+",
                        default=[1, 3, 5, 6, 7],
                        help="Layer counts to prune and evaluate")
    parser.add_argument("--lm_head_type", type=str, default="frozen",
                        choices=["frozen", "trained"],
                        help="SSN only. 'frozen': use model's own lm_head (default). "
                             "'trained': load per-layer lm_head checkpoints "
                             "(requires --lm_head_ckpt_dir).")
    parser.add_argument("--lm_head_ckpt_dir", type=str, default=None,
                        help="Directory of per-layer lm_head checkpoints "
                             "(required when --lm_head_type trained).")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging.")
    return parser.parse_args()


def _model_tag(model_name: str) -> str:
    """Derive a filesystem-safe short name from a HuggingFace model ID or path."""
    return model_name.rstrip("/").split("/")[-1]


def main():
    args = parse_args()
    abbr = MEASURE_ABBR[args.measure]
    model_tag = _model_tag(args.model_name)
    mode_tag = args.mode[0].upper()

    # ------------------------------------------------------------------
    # wandb init
    # ------------------------------------------------------------------
    if args.wandb:
        import wandb
        wandb.init(
            project="tabp",
            name=f"{args.method}_{abbr}_{model_tag}_{args.mode}",
            config=vars(args),
        )

    # ------------------------------------------------------------------
    # Step 1: Load dataset, model, and rank layers
    # ------------------------------------------------------------------
    print(f"\n[1/2] Loading {args.model_name} and ranking layers "
          f"(method={args.method}, measure={args.measure}, mode={args.mode}) ...")

    model, tokenizer = load_model(args.model_name)

    dataset, split, task_type, prepare_input = load_calib_dataset(
        args.calib_data, args.dataset_path
    )

    if args.method == "ssn":
        sorted_blocks = rank_blocks_ssn(
            model=model,
            tokenizer=tokenizer,
            prefix=PREFIX,
            postfix=POSTFIX,
            mode=args.mode,
            measure=args.measure,
            dataset=dataset,
            split=split,
            task_type=task_type,
            prepare_input=prepare_input,
            n_samples=args.n_samples,
            lm_head_type=args.lm_head_type,
            lm_head_ckpt_dir=args.lm_head_ckpt_dir,
            use_wandb=args.wandb,
        )
    else:  # ddf
        sorted_blocks = rank_blocks_ddf(
            model=model,
            tokenizer=tokenizer,
            mode=args.mode,
            measure=args.measure,
            dataset=dataset,
            split=split,
            task_type=task_type,
            prepare_input=prepare_input,
            prefix=PREFIX,
            postfix=POSTFIX,
            n_samples=args.n_samples,
            n_windows=args.n_windows,
            n_steps=args.n_steps,
            use_wandb=args.wandb,
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if args.wandb:
        import wandb
        wandb.log({"ranking/sorted_blocks": wandb.Table(
            columns=["rank", "block_idx"],
            data=[[i, int(b)] for i, b in enumerate(sorted_blocks)],
        )})

    # ------------------------------------------------------------------
    # Step 2: Iterative pruning & evaluation
    # ------------------------------------------------------------------
    method_tag = args.method.upper()
    print(f"\n[2/2] Evaluating pruning counts: {args.prune_counts} ...")
    for n_prune in args.prune_counts:
        layers_to_prune = sorted_blocks[:n_prune]
        print(f"\nPruned Blocks: {layers_to_prune.tolist()}")

        score_dir = os.path.join(
            "results", f"{method_tag}_{abbr}_{model_tag}{mode_tag}", str(n_prune)
        )
        os.makedirs(score_dir, exist_ok=True)

        model, tokenizer = load_model(args.model_name)
        prune_layers(model, layers_to_prune)

        # Perplexity (wikitext2 + ptb)
        if not (os.path.exists(os.path.join(score_dir, "ppl.csv"))
                or os.path.exists(os.path.join(score_dir, "ppl_bos.csv"))):
            for add_bos in [True, False]:
                eval_ppl(
                    output_dir=score_dir,
                    model=model,
                    tokenizer=tokenizer,
                    datasets=["wikitext2"],
                    add_bos_to_every=add_bos,
                )

        # Text generation sample
        if not os.path.exists(os.path.join(score_dir, "gen_text.txt")):
            generate_txt(output_dir=score_dir, model=model,
                         tokenizer=tokenizer, device="cuda")

        # Zero-shot accuracy on 7 benchmarks
        if not os.path.exists(os.path.join(score_dir, "zeroshot_acc.csv")):
            eval_acc(
                model=model,
                tasks_list=",".join(TASK_LIST),
                batch_size=1,
                max_batch_size=1,
                output_json=os.path.join(score_dir, "zeroshot_acc.json"),
                limit_fixed=99999999,
                no_cache=False,
                write_out=True,
                output_base_path=score_dir,
            )

        if args.wandb:
            import wandb
            wandb_metrics = {"eval/n_prune": n_prune}
            ppl_path = os.path.join(score_dir, "ppl.csv")
            if os.path.exists(ppl_path):
                import csv
                with open(ppl_path) as f:
                    rows = list(csv.reader(f))
                    for k, v in zip(rows[0], rows[1]):
                        try:
                            wandb_metrics[f"eval/{k}"] = float(v)
                        except ValueError:
                            pass
            acc_path = os.path.join(score_dir, "zeroshot_acc.json")
            if os.path.exists(acc_path):
                with open(acc_path) as f:
                    acc_data = json.load(f)
                for task, res in acc_data.get("results", {}).items():
                    for metric, val in res.items():
                        if isinstance(val, (int, float)):
                            wandb_metrics[f"eval/{task}/{metric}"] = val
            wandb.log(wandb_metrics)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
