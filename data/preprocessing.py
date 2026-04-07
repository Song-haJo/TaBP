"""
Dataset loading and preprocessing utilities for TaBP.

Handles:
    - Loading calibration datasets from HuggingFace Hub or local disk
    - Formatting dataset samples into model inputs
    - Registering dataset-specific prepare_input functions

To add a new dataset:
    1. Write a prepare_input(example, tokenizer, prefix, postfix) function.
       - For QA datasets: return (input_str, logits_processor, allowed_token_ids, key_token_id)
       - For text generation datasets: return the text string directly (or None to skip)
    2. Register it in _DATASET_REGISTRY with the correct task_type.
"""

from typing import Optional, Callable, Tuple

from datasets import load_from_disk

from models.loader import ForcedTokenLogitsProcessor


def prepare_arc_inputs(example, tokenizer, prefix: str, postfix: str):
    """
    Format one AI2 ARC-Easy sample into a QA prompt with answer constraints.

    Args:
        example:   Dataset row with keys 'question', 'choices', 'answerKey'.
        tokenizer: The model's tokenizer.
        prefix:    Instruction text prepended to the question.
        postfix:   Text appended after the choices (e.g., "Answer: ").

    Returns:
        inputs (str)              – Formatted prompt string.
        logits_processor          – ForcedTokenLogitsProcessor for answer tokens.
        allowed_token_ids (list)  – Token IDs of the answer choices.
        key_token_id (int)        – Token ID of the correct answer.
    """
    question = example["question"]
    choices = example["choices"]
    label = example["answerKey"]

    allowed_tokens = choices["label"]
    allowed_token_ids = tokenizer.convert_tokens_to_ids(allowed_tokens)
    logits_processor = ForcedTokenLogitsProcessor(allowed_token_ids)

    options = "".join(
        f"{lbl}. {text}\n"
        for lbl, text in zip(choices["label"], choices["text"])
    )
    inputs = prefix + "Question: " + question + "\n" + options + postfix
    key_token_id = tokenizer.convert_tokens_to_ids(label)

    return inputs, logits_processor, allowed_token_ids, key_token_id


def _prepare_wikitext(example, tokenizer, prefix: str, postfix: str):
    """Return the raw text from a WikiText sample (prefix/postfix unused)."""
    return example.get("text", "").strip()


# Registry: dataset name → {hf_name, hf_config, split, task_type, prepare_input}
# task_type: "qa"              → one forward pass per sample, forced answer tokens
#            "text_generation" → sliding window + autoregressive generation
_DATASET_REGISTRY = {
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


def load_calib_dataset(
    calib_data: str,
    dataset_path: Optional[str] = None,
) -> Tuple:
    """
    Load a calibration dataset and return everything rank_blocks needs.

    Args:
        calib_data:   Key in _DATASET_REGISTRY (e.g. 'arc_easy', 'wikitext').
        dataset_path: Local path (load_from_disk). None → HuggingFace Hub.

    Returns:
        (dataset, split, task_type, prepare_input)
          dataset       – HuggingFace DatasetDict
          split         – Split name string (e.g. 'train')
          task_type     – 'qa' or 'text_generation'
          prepare_input – Callable(example, tokenizer, prefix, postfix)
    """
    if calib_data not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown calib_data: '{calib_data}'. "
            f"Choose from {list(_DATASET_REGISTRY)}"
        )
    meta = _DATASET_REGISTRY[calib_data]

    if dataset_path is not None:
        dataset = load_from_disk(dataset_path)
    else:
        from datasets import load_dataset
        dataset = load_dataset(meta["hf_name"], meta["hf_config"])

    return dataset, meta["split"], meta["task_type"], meta["prepare_input"]
