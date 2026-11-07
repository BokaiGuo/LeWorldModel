"""Generic paired-bank construction; planner receives queries, never dense truth."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from experiments.pc_cross_query.core import Ledger as BaseLedger, RCOND

MATCHED = ("cross_interaction", "additive", "parent", "row_mean",
           "probe_min", "shuffled", "random_missing")
A = np.arange(4)


class Ledger(BaseLedger):
    def query(self, actions, stage, pairs=None):
        started = time.perf_counter()
        values = super().query(actions, stage, pairs)
        self.trace[-1]["query_wall_seconds"] = time.perf_counter() - started
        return values


def integer_pairs(value, shape):
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != 2 or not len(raw):
        raise ValueError("nonempty [row,column] pairs required")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("integer action identities required")
    if (raw < 0).any() or (raw >= shape).any():
        raise ValueError("action identity outside bank")
    return raw.astype(int)


@dataclass(frozen=True)
class PairedBank:
    """Complete legal nominal witnesses are retained before any recombination."""
    nominals: np.ndarray
    prefix_steps: int = 5

    def __post_init__(self):
        values = np.array(self.nominals, dtype=np.float64, copy=True)
        if values.ndim != 3 or values.shape[1:] != (15, 2) or len(values) < 4:
            raise ValueError("at least four complete 15-step nominal chains required")
        if not np.isfinite(values).all() or np.max(np.abs(values)) > 1:
            raise ValueError("illegal nominal actions")
        if self.prefix_steps != 5:
            raise ValueError("this protocol requires five-step prefixes")
        values.flags.writeable = False
        object.__setattr__(self, "nominals", values)

    @property
    def shape(self):
        return len(self.nominals), len(self.nominals) + 1

    @property
    def lineage(self):
        return np.column_stack((np.arange(len(self.nominals)), np.arange(1, len(self.nominals) + 1)))

    def compose(self, pairs):
        pairs = integer_pairs(pairs, self.shape)
        prefixes = self.nominals[pairs[:, 0], :5]
        tails = np.concatenate((np.zeros((1, 10, 2)), self.nominals[:, 5:]))
        return np.concatenate((prefixes, tails[pairs[:, 1]]), axis=1)


def anchors(a_cost):
    a_cost = np.asarray(a_cost, float)
    if a_cost.ndim != 2 or a_cost.shape[1] != 4 or len(a_cost) < 4 or not np.isfinite(a_cost).all():
        raise ValueError("finite N-by-four first-stage costs required")
    first = int(np.argmin(a_cost.min(1)))
    residual = a_cost - a_cost[:, :1]
    residual -= residual[first]
    chosen, directions = [first], []
    for _ in range(2):
        remaining = residual.copy()
        for d in directions:
            remaining -= (remaining @ d)[:, None] * d
        norms = (remaining ** 2).sum(1)
        norms[chosen] = -np.inf
        index = int(np.argmax(norms))
        chosen.append(index)
        norm = np.linalg.norm(remaining[index])
        if norm > 1e-12:
            directions.append(remaining[index] / norm)
    return np.asarray(chosen)


def support(shape, chosen):
    n, m = shape
    chosen = np.asarray(chosen, int)
    if m != n + 1 or chosen.shape != (3,) or len(np.unique(chosen)) != 3 or np.any((chosen < 0) | (chosen >= n)):
        raise ValueError("paired rectangular bank and three distinct anchors required")
    mask = np.zeros(shape, bool)
    mask[:, A] = True
    mask[chosen] = True
    mask[np.arange(n), np.arange(n) + 1] = True
    return mask


def check_observed(observed, chosen):
    costs = np.asarray(observed, float)
    mask = support(costs.shape, chosen)
    if not np.isfinite(costs[mask]).all() or not np.isnan(costs[~mask]).all():
        raise ValueError("unknown cells must be NaN; no evaluator truth allowed")
    return costs, mask


def interaction(observed, chosen):
    costs, mask = check_observed(observed, chosen)
    base = costs[:, :1] + (costs[chosen[0]] - costs[chosen[0], 0])[None]
    columns = costs[:, A] - base[:, A]
    rows = costs[chosen] - base[chosen]
    intersection = columns[chosen]
    result = base + columns @ np.linalg.pinv(intersection, rcond=RCOND) @ rows
    singular = np.linalg.svd(intersection, compute_uv=False)
    rank = np.count_nonzero(singular > RCOND * singular[0]) if singular[0] else 0
    result[mask] = costs[mask]
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite interaction prediction")
    return result, {"retained_rank": int(rank), "singular_values": singular.tolist()}


def prediction(arm, observed, chosen, seed):
    costs, mask = check_observed(observed, chosen)
    n, m = costs.shape
    diagnostic = {}
    if arm == "cross_interaction":
        result, diagnostic = interaction(costs, chosen)
    elif arm == "shuffled":
        shuffled = costs.copy()
        rng = np.random.default_rng(seed)
        for i in chosen:
            movable = np.array([j for j in range(4, m) if j != i + 1], int)
            shuffled[i, movable] = costs[i, movable[rng.permutation(len(movable))]]
        result, diagnostic = interaction(shuffled, chosen)
        diagnostic["shuffled_observed"] = [
            [float(x) if np.isfinite(x) else None for x in row] for row in shuffled
        ]
    elif arm == "additive":
        pairs = np.argwhere(mask)
        design = np.zeros((len(pairs), n + m))
        design[np.arange(len(pairs)), pairs[:, 0]] = 1.
        design[np.arange(len(pairs)), n + pairs[:, 1]] = 1.
        beta = np.linalg.lstsq(design, costs[mask], rcond=RCOND)[0]
        result = beta[:n, None] + beta[None, n:]
    elif arm in ("parent", "row_mean", "probe_min"):
        value = (costs[np.arange(n), np.arange(n) + 1] if arm == "parent"
                 else costs[:, :4].mean(1) if arm == "row_mean" else costs[:, :4].min(1))
        result = np.broadcast_to(value[:, None], costs.shape).copy()
    else:
        raise ValueError("unknown deterministic allocation arm")
    result[mask] = costs[mask]
    return result, diagnostic


def choose(predicted, mask, count):
    if predicted.shape != mask.shape or not np.isfinite(predicted).all():
        raise ValueError("finite prediction with matching shape required")
    remaining = np.flatnonzero(~mask.ravel())
    if not 0 < count <= len(remaining):
        raise ValueError("invalid remaining budget")
    ids = remaining[np.argsort(predicted.ravel()[remaining], kind="stable")[:count]]
    return np.column_stack(np.unravel_index(ids, mask.shape))


def acquire(bank, ledger):
    observed = np.full(bank.shape, np.nan)

    def query_new(pairs, stage):
        pairs = integer_pairs(pairs, bank.shape)
        needed = pairs[np.isnan(observed[pairs[:, 0], pairs[:, 1]])]
        if len(np.unique(needed, axis=0)) != len(needed):
            raise ValueError("duplicate requested pairs")
        if len(needed):
            costs = ledger.query(bank.compose(needed), stage, needed)
            observed[needed[:, 0], needed[:, 1]] = costs

    query_new(bank.lineage, "paid_nominal_witnesses")
    query_new(np.asarray([(i, j) for i in range(bank.shape[0]) for j in A]), "cross_A")
    selected = anchors(observed[:, :4])
    query_new(np.asarray([(i, j) for i in selected for j in range(bank.shape[1])]), "cross_rows")
    check_observed(observed, selected)
    if ledger.count != int(np.isfinite(observed).sum()):
        raise ValueError("duplicate query accounting")
    return observed, selected
