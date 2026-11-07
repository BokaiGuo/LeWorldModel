"""Only current-task queried costs enter the crossed-query allocator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

A_COLUMNS = np.arange(4)
MATCHED_ARMS = (
    "cross_interaction", "additive", "row_mean", "probe_min",
    "shuffled", "random_missing",
)
RCOND = 1e-8


def validate_bank(prefixes, tails):
    prefixes, tails = np.asarray(prefixes, float), np.asarray(tails, float)
    if prefixes.shape != (8, 5, 2) or tails.shape != (32, 10, 2):
        raise ValueError("expected 8 five-step prefixes and 32 ten-step tails")
    if not all(np.isfinite(x).all() and np.max(np.abs(x)) <= 1 for x in (prefixes, tails)):
        raise ValueError("nonfinite or out-of-bounds action")
    if np.any(tails[0]):
        raise ValueError("column zero must be declared same-duration hold")
    return prefixes, tails


def compose(prefixes, tails, pairs):
    prefixes, tails = validate_bank(prefixes, tails)
    pairs = np.asarray(pairs, int)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs):
        raise ValueError("nonempty row-column pairs required")
    if (pairs < 0).any() or (pairs >= [8, 32]).any():
        raise ValueError("invalid action identity")
    return np.concatenate((prefixes[pairs[:, 0]], tails[pairs[:, 1]]), axis=1)


def anchor_rows(a_cost):
    a_cost = np.asarray(a_cost, float)
    if a_cost.shape != (8, 4) or not np.isfinite(a_cost).all():
        raise ValueError("expected finite first-stage query costs")
    first = int(np.argmin(a_cost.min(axis=1)))
    effects = a_cost - a_cost[:, :1]
    residual = effects - effects[first]
    anchors, directions = [first], []
    for _ in range(2):
        remaining = residual.copy()
        for direction in directions:
            remaining -= (remaining @ direction)[:, None] * direction
        norms = np.sum(remaining ** 2, axis=1)
        norms[anchors] = -np.inf
        chosen = int(np.argmax(norms))
        anchors.append(chosen)
        norm = np.linalg.norm(remaining[chosen])
        if norm > 1e-12:
            directions.append(remaining[chosen] / norm)
    return np.asarray(anchors, int)


def cross_mask(anchors):
    anchors = np.asarray(anchors, int)
    if anchors.shape != (3,) or len(np.unique(anchors)) != 3 or np.any((anchors < 0) | (anchors >= 8)):
        raise ValueError("three distinct anchor rows required")
    mask = np.zeros((8, 32), bool)
    mask[:, A_COLUMNS] = True
    mask[anchors] = True
    return mask


def validate_observed(observed, anchors):
    observed = np.asarray(observed, float)
    mask = cross_mask(anchors)
    if observed.shape != mask.shape:
        raise ValueError("expected 8x32 costs")
    if not np.isfinite(observed[mask]).all() or not np.isnan(observed[~mask]).all():
        raise ValueError("observed costs required; unknown cells must remain NaN")
    return observed, mask


def interaction_prediction(observed, anchors):
    observed, mask = validate_observed(observed, anchors)
    anchors = np.asarray(anchors)
    first = int(anchors[0])
    col_effect = observed[first] - observed[first, 0]
    baseline = observed[:, :1] + col_effect[None]
    da = observed[:, A_COLUMNS] - baseline[:, A_COLUMNS]
    di = observed[anchors] - baseline[anchors]
    intersection = da[anchors]
    prediction = baseline + da @ np.linalg.pinv(intersection, rcond=RCOND) @ di
    singular = np.linalg.svd(intersection, compute_uv=False)
    threshold = RCOND * singular[0] if singular[0] else 0.
    prediction[mask] = observed[mask]
    if not np.isfinite(prediction).all():
        raise FloatingPointError("nonfinite interaction approximation")
    return prediction, {
        "singular_values": singular.tolist(),
        "retained_rank": int(np.count_nonzero(singular > threshold)),
        "rcond": RCOND,
    }


def additive_prediction(observed, anchors):
    observed, mask = validate_observed(observed, anchors)
    pairs = np.argwhere(mask)
    design = np.zeros((len(pairs), 8 + 32))
    design[np.arange(len(pairs)), pairs[:, 0]] = 1.
    design[np.arange(len(pairs)), 8 + pairs[:, 1]] = 1.
    coefficients = np.linalg.lstsq(design, observed[mask], rcond=RCOND)[0]
    result = coefficients[:8, None] + coefficients[None, 8:]
    result[mask] = observed[mask]
    return result


def shuffle_identity(observed, anchors, seed):
    observed, _ = validate_observed(observed, anchors)
    changed = observed.copy()
    rng = np.random.default_rng(seed)
    # Do not shuffle A or the hold reference. Preserve each anchor row's
    # collection of measured B costs, while breaking their action identities.
    for row in anchors:
        changed[row, 4:] = observed[row, 4:][rng.permutation(28)]
    return changed


def predict_all(observed, anchors, seed):
    observed, mask = validate_observed(observed, anchors)
    cross, diagnostics = interaction_prediction(observed, anchors)
    shuffled, _ = interaction_prediction(shuffle_identity(observed, anchors, seed), anchors)
    values = {
        "cross_interaction": cross,
        "additive": additive_prediction(observed, anchors),
        "row_mean": np.broadcast_to(observed[:, :4].mean(axis=1, keepdims=True), (8, 32)).copy(),
        "probe_min": np.broadcast_to(observed[:, :4].min(axis=1, keepdims=True), (8, 32)).copy(),
        "shuffled": shuffled,
    }
    for value in values.values():
        value[mask] = observed[mask]
    return values, diagnostics


def proposals(prediction, observed_mask, count=12):
    prediction = np.asarray(prediction, float)
    mask = np.asarray(observed_mask, bool)
    if prediction.shape != (8, 32) or mask.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("finite 8x32 prediction required")
    missing = np.flatnonzero(~mask.ravel())
    if not 0 < count <= len(missing):
        raise ValueError("proposal count exceeds missing cells")
    chosen = missing[np.argsort(prediction.ravel()[missing], kind="stable")[:count]]
    return np.column_stack(np.unravel_index(chosen, (8, 32)))


@dataclass
class Ledger:
    """Budgeted model callback; never accepts evaluator truth or a dense pool."""

    evaluate: Callable
    budget: int
    trace: list = field(default_factory=list)
    count: int = 0

    def query(self, actions, stage, pairs=None):
        actions = np.asarray(actions, float)
        if actions.ndim != 3 or actions.shape[1:] != (15, 2) or not len(actions):
            raise ValueError("full 15-step chains required")
        if not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1:
            raise ValueError("illegal query")
        if self.count + len(actions) > self.budget:
            raise ValueError("query budget exhausted")
        costs, endpoints, seconds = self.evaluate(actions.copy())
        costs, endpoints = np.asarray(costs, float), np.asarray(endpoints, float)
        if costs.shape != (len(actions),) or not np.isfinite(costs).all():
            raise ValueError("invalid model costs")
        if endpoints.shape != (len(actions), 2) or not np.isfinite(endpoints).all():
            raise ValueError("invalid model endpoints")
        self.count += len(actions)
        self.trace.append({
            "stage": stage, "pairs": None if pairs is None else np.asarray(pairs).tolist(),
            "actions": actions.tolist(), "costs": costs.tolist(),
            "endpoints": endpoints.tolist(), "model_seconds": float(seconds),
        })
        return costs

    def best(self):
        if not self.trace:
            raise ValueError("no actually queried chain")
        costs = np.concatenate([batch["costs"] for batch in self.trace])
        actions = np.concatenate([batch["actions"] for batch in self.trace])
        index = int(np.argmin(costs))
        return {"query_index": index, "score": float(costs[index]), "chain": actions[index].tolist()}


def acquire_cross(prefixes, tails, ledger):
    first_pairs = np.asarray([(i, j) for i in range(8) for j in A_COLUMNS])
    a = ledger.query(compose(prefixes, tails, first_pairs), "cross_A", first_pairs).reshape(8, 4)
    anchors = anchor_rows(a)
    remaining = np.asarray([(int(i), j) for i in anchors for j in range(4, 32)])
    rest = ledger.query(compose(prefixes, tails, remaining), "cross_rows", remaining)
    observed = np.full((8, 32), np.nan)
    observed[:, :4] = a
    observed[remaining[:, 0], remaining[:, 1]] = rest
    validate_observed(observed, anchors)
    return observed, anchors
