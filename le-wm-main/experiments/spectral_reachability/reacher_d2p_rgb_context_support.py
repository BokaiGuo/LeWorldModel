"""Training-free RGB-history-conditioned official packet support.

Only source episode metadata, RGB histories and next actions enter the index.
No qpos, future observation, goal label or evaluator outcome enters retrieval.
Random and retrieved proposals draw from the SAME frozen packet index.
"""
from __future__ import annotations

import numpy as np
import torch


def source_anchors(lengths, offsets, *, excluded, count, seed):
    lengths, offsets = np.asarray(lengths, np.int64), np.asarray(offsets, np.int64)
    if lengths.ndim != 1 or lengths.shape != offsets.shape or count < 1:
        raise ValueError("matching source metadata and positive index size required")
    eligible = np.array([int(index) for index in np.flatnonzero(lengths >= 21) if int(index) not in excluded], np.int64)
    if len(eligible) < count:
        raise ValueError("insufficient episode-disjoint index anchors; no replacement allowed")
    rng = np.random.default_rng(seed)
    episodes = rng.choice(eligible, count, replace=False)
    anchors = [{"source_episode": int(episode), "root_row": int(offsets[episode]) + int(rng.integers(15, lengths[episode] - 5))}
        for episode in episodes]
    return anchors


def nearest_context_indices(query, index, *, count):
    if query.ndim != 2 or index.ndim != 3 or tuple(index.shape[1:]) != tuple(query.shape) or count < 1 or count > len(index):
        raise ValueError("matching temporal RGB features and valid proposal count required")
    if index.device != query.device or not torch.isfinite(index).all() or not torch.isfinite(query).all():
        raise ValueError("finite same-device context features required")
    # Preserve the full ordered history: equal last frames need not imply equal velocity.
    distances = (index - query[None]).square().mean(dim=(1, 2))
    return torch.argsort(distances, stable=True)[:count], distances


def ranking_metrics(predicted, observed):
    predicted, observed = np.asarray(predicted, np.float64), np.asarray(observed, np.float64)
    if predicted.ndim != 1 or observed.shape != predicted.shape or len(predicted) < 2 or not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError("finite matched candidate costs with zero control first required")
    left, right = np.triu_indices(len(predicted), k=1)
    product = (predicted[left] - predicted[right]) * (observed[left] - observed[right])
    pm, om = predicted[1:] - predicted[0], observed[1:] - observed[0]
    non_greedy = pm >= 1e-6
    count = int(non_greedy.sum())
    return {"candidates": len(predicted), "mean_abs_cost_error": float(np.abs(predicted - observed).mean()),
        "mean_abs_zero_relative_margin_error": float(np.abs(pm - om).mean()),
        "pairwise_strict_order_agreements": int((product > 0).sum()),
        "pairwise_strict_order_reversals": int((product < 0).sum()), "pairwise_ties": int((product == 0).sum()),
        "pairwise_agreement_fraction": float((product > 0).mean()),
        "predicted_non_greedy_count": count,
        "observed_non_greedy_among_predicted": int((non_greedy & (om >= 1e-6)).sum()),
        "non_greedy_precision": float((om[non_greedy] >= 1e-6).mean()) if count else None}
