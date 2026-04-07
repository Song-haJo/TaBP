"""
Layer-importance metrics for FluctARCE (logit-based).

Applied to the lm_head output distribution at each layer:
    Single-distribution : compute_entropy, compute_confidence, compute_gap
    Answer-key          : compute_answerkey_prob
    Distribution-pair   : compute_cross_entropy, compute_kl_divergence,
                          compute_js_divergence, compute_earth_movers_distance,
                          compute_total_variation_distance,
                          compute_hellinger_distance, compute_cosine_similarity,
                          compute_bhattacharyya_distance, compute_energy_distance
"""

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import norm
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance


# ---------------------------------------------------------------------------
# Single-distribution metrics
# ---------------------------------------------------------------------------

def compute_entropy(probs) -> float:
    """Shannon entropy of a probability distribution."""
    return -np.sum(probs * np.log(probs + 1e-9))


def compute_confidence(probs) -> float:
    """Maximum probability (model confidence)."""
    return float(np.max(probs))


def compute_gap(probs) -> float:
    """Difference between the top-1 and top-2 probabilities."""
    sorted_probs = np.sort(probs)
    return float(sorted_probs[-1] - sorted_probs[-2])


def compute_answerkey_prob(probs, correct_token_id: int, allowed_token_ids: list) -> float:
    """Probability assigned to the correct answer token."""
    correct_index = allowed_token_ids.index(correct_token_id)
    return float(probs[correct_index])


# ---------------------------------------------------------------------------
# Distribution-comparison metrics
# ---------------------------------------------------------------------------

def compute_cross_entropy(logits, target_probs, allowed_token_ids) -> float:
    """
    Cross-entropy of the target distribution (final layer) relative to
    the layer's logits, restricted to allowed tokens.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = torch.tensor(logits, dtype=torch.float32).unsqueeze(0).to(device)
    target_probs = torch.tensor(target_probs, dtype=torch.float32).unsqueeze(0).to(device)
    allowed_ids = torch.tensor(allowed_token_ids, dtype=torch.long).to(device)

    logits = logits[:, allowed_ids]
    target_probs = target_probs[:, allowed_ids]

    log_probs = F.log_softmax(logits, dim=-1)
    return (-torch.sum(target_probs * log_probs, dim=-1).mean()).item()


def compute_kl_divergence(P, Q, allowed_token_ids, epsilon: float = 1e-10) -> float:
    """KL divergence D_KL(P || Q), with P filtered by allowed_token_ids."""
    if isinstance(P, np.ndarray):
        P = torch.tensor(P, dtype=torch.float32)
    if isinstance(Q, np.ndarray):
        Q = torch.tensor(Q, dtype=torch.float32)

    P = P[allowed_token_ids] + epsilon
    Q = Q + epsilon
    return torch.sum(P * torch.log(P / Q), dim=-1).mean().item()


def compute_js_divergence(P, Q, allowed_token_ids, epsilon: float = 1e-8) -> float:
    """Jensen-Shannon divergence between P and Q over allowed tokens."""
    if isinstance(P, np.ndarray):
        P = torch.tensor(P, dtype=torch.float32)
    if isinstance(Q, np.ndarray):
        Q = torch.tensor(Q, dtype=torch.float32)

    P = P[allowed_token_ids]
    Q = Q[allowed_token_ids]
    P = P / P.sum()
    Q = Q / Q.sum()
    M = 0.5 * (P + Q)

    js = 0.5 * (
        torch.sum(P * torch.log((P + epsilon) / (M + epsilon)))
        + torch.sum(Q * torch.log((Q + epsilon) / (M + epsilon)))
    )
    return js.item()


def compute_earth_movers_distance(P, Q, allowed_token_ids) -> float:
    """Earth Mover's Distance (Wasserstein-1) over allowed tokens."""
    P = np.array(P)[allowed_token_ids]
    Q = np.array(Q)[allowed_token_ids]
    P = P / P.sum()
    Q = Q / Q.sum()
    return float(wasserstein_distance(P, Q))


def compute_total_variation_distance(P, Q, allowed_token_ids) -> float:
    """Total Variation Distance over allowed tokens."""
    P = np.array(P)[allowed_token_ids]
    Q = np.array(Q)[allowed_token_ids]
    P = P / P.sum()
    Q = Q / Q.sum()
    return float(0.5 * np.sum(np.abs(P - Q)))


def compute_hellinger_distance(P, Q, allowed_token_ids) -> float:
    """Hellinger distance over allowed tokens."""
    P = np.array(P)[allowed_token_ids]
    Q = np.array(Q)[allowed_token_ids]
    P = P / P.sum()
    Q = Q / Q.sum()
    return float(np.sqrt(0.5 * np.sum((np.sqrt(P) - np.sqrt(Q)) ** 2)))


def compute_cosine_similarity(P, Q, allowed_token_ids) -> float:
    """Cosine similarity over allowed tokens."""
    P = np.array(P)[allowed_token_ids]
    Q = np.array(Q)[allowed_token_ids]
    P = P / P.sum()
    Q = Q / Q.sum()
    return float(np.dot(P, Q) / (norm(P) * norm(Q)))


def compute_bhattacharyya_distance(P, Q, allowed_token_ids) -> float:
    """Bhattacharyya distance over allowed tokens."""
    P = np.array(P)[allowed_token_ids]
    Q = np.array(Q)[allowed_token_ids]
    P = P / P.sum()
    Q = Q / Q.sum()
    bc = np.sum(np.sqrt(P * Q))
    return float(-np.log(bc + 1e-10))


def compute_energy_distance(P, Q, allowed_token_ids) -> float:
    """Energy distance over allowed tokens."""
    P = np.array(P)[allowed_token_ids]
    Q = np.array(Q)[allowed_token_ids]
    P = P / P.sum()
    Q = Q / Q.sum()
    return float(np.mean([euclidean(p, q) for p, q in zip(P, Q)]))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

MEASURE_ABBR = {
    "entropy":            "ENT",
    "confidence_score":   "CONF",
    "gap":                "G",
    "key":                "K",
    "cross_entropy":      "CE",
    "kl_divergence":      "KLD",
    "js_divergence":      "JSD",
    "wasserstein_dist":   "WD",
    "hellinger_dist":     "HD",
    "bhattacharyya_dist": "BD",
    "cosine_similarity":  "COS",
    "tvd":                "TVD",
    "energy":             "ED",
}

# α coefficient for DDF: +1 if higher values are desirable, -1 if lower is desirable.
# A desirable shift at block i means α * (s_i - s_{i-1}) > 0.
MEASURE_DIRECTION = {
    "entropy":            -1,  # low entropy = more certain → decrease is desirable
    "confidence_score":   +1,  # high confidence → increase is desirable
    "gap":                +1,  # large gap → increase is desirable
    "key":                +1,  # high correct-answer prob → increase is desirable
    "cross_entropy":      -1,  # low CE → decrease is desirable
    "kl_divergence":      -1,  # low KLD → decrease is desirable
    "js_divergence":      -1,  # low JSD → decrease is desirable
    "wasserstein_dist":   -1,  # low WD → decrease is desirable
    "hellinger_dist":     -1,  # low HD → decrease is desirable
    "bhattacharyya_dist": -1,  # low BD → decrease is desirable
    "cosine_similarity":  +1,  # high cosine sim → increase is desirable
    "tvd":                -1,  # low TVD → decrease is desirable
    "energy":             -1,  # low ED → decrease is desirable
}


def compute_metric(measure: str, layer_probs, target_probs, logits,
                   allowed_token_ids, key_token_id) -> float:
    """
    Dispatch to the appropriate logit-based metric function.

    Args:
        measure:           Metric name (see MEASURE_ABBR for valid keys).
        layer_probs:       Softmax probs from intermediate layer
                           (restricted to allowed_token_ids).
        target_probs:      Full softmax probs from the final layer.
        logits:            Full softmax probs from the intermediate layer
                           (unrestricted; used for cross_entropy).
        allowed_token_ids: Token IDs of the answer choices.
        key_token_id:      Token ID of the correct answer.

    Returns:
        Scalar metric value.
    """
    if measure == "entropy":
        return compute_entropy(layer_probs)
    elif measure == "confidence_score":
        return compute_confidence(layer_probs)
    elif measure == "gap":
        return compute_gap(layer_probs / layer_probs.sum())
    elif measure == "key":
        return compute_answerkey_prob(layer_probs, key_token_id, allowed_token_ids)
    elif measure == "cross_entropy":
        return compute_cross_entropy(logits, target_probs, allowed_token_ids)
    elif measure == "kl_divergence":
        return compute_kl_divergence(target_probs, layer_probs, allowed_token_ids)
    elif measure == "js_divergence":
        return compute_js_divergence(target_probs, layer_probs, allowed_token_ids)
    elif measure == "wasserstein_dist":
        return compute_earth_movers_distance(target_probs, layer_probs, allowed_token_ids)
    elif measure == "tvd":
        return compute_total_variation_distance(target_probs, layer_probs, allowed_token_ids)
    elif measure == "hellinger_dist":
        return compute_hellinger_distance(target_probs, layer_probs, allowed_token_ids)
    elif measure == "cosine_similarity":
        return compute_cosine_similarity(target_probs, layer_probs, allowed_token_ids)
    elif measure == "bhattacharyya_dist":
        return compute_bhattacharyya_distance(target_probs, layer_probs, allowed_token_ids)
    elif measure == "energy":
        return compute_energy_distance(target_probs, layer_probs, allowed_token_ids)
    else:
        raise ValueError(f"Unknown measure: '{measure}'. "
                         f"Valid options: {list(MEASURE_ABBR)}")
