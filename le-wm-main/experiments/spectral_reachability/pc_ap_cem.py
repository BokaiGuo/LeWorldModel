"""AP paper's single-block CEM contract; predictor is supplied by caller.

No model or simulator is included. A production caller must bind a frozen
CUDA predictor, correct action normalizer and real feedback independently.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from experiments.spectral_reachability.pc_ap_observation_memory import ObservationMemory, Retrieval


@dataclass(frozen=True)
class CEMResult:
    normalized_block: np.ndarray
    score: float
    predicted_blocks: int
    prediction_batches: int


@dataclass(frozen=True)
class APDecision:
    retrieval: Retrieval
    plan: CEMResult


def plan_ap_decision(
    memory: ObservationMemory,
    current: np.ndarray,
    goal: np.ndarray,
    goal_offset: int,
    elapsed: int,
    predict_endpoints: Callable[[np.ndarray], np.ndarray],
    low: np.ndarray,
    high: np.ndarray,
    **cem_options,
) -> APDecision:
    """Select one block using an observed successor and a caller-bound predictor."""
    if np.asarray(low).ndim != 2 or np.asarray(low).shape[0] != memory.block:
        raise ValueError("CEM block length must match observed successor offset")
    retrieval = memory.retrieve(current, goal, goal_offset, elapsed)

    def evaluate(actions):
        endpoints = np.asarray(predict_endpoints(actions), dtype=np.float64)
        if endpoints.shape != (len(actions), retrieval.target.size) or not np.isfinite(endpoints).all():
            raise ValueError("predictor endpoint shape or values invalid")
        return ((endpoints - retrieval.target) ** 2).sum(axis=1)

    return APDecision(retrieval, plan_ap_cem(evaluate, low, high, **cem_options))


def plan_ap_cem(
    evaluate: Callable[[np.ndarray], np.ndarray],
    low: np.ndarray,
    high: np.ndarray,
    *,
    seed: int,
    population: int = 300,
    iterations: int = 30,
    elites: int = 30,
) -> CEMResult:
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    if (low.ndim != 2 or low.shape != high.shape or low.size == 0
            or not np.isfinite(low).all() or not np.isfinite(high).all()
            or np.any(low >= high)):
        raise ValueError("invalid normalized action bounds")
    if (not all(isinstance(x, int) for x in (population, iterations, elites))
            or population < 1 or iterations < 1 or not 1 <= elites <= population):
        raise ValueError("invalid CEM budget")
    rng = np.random.default_rng(seed)
    initial = np.zeros_like(low)
    mean = initial.copy()
    std = np.full_like(low, 1 / 3)

    def score(candidates):
        values = np.asarray(evaluate(candidates), dtype=np.float64)
        if values.shape != (len(candidates),) or not np.isfinite(values).all():
            raise ValueError("predictor must return one finite score per candidate")
        return values

    for _ in range(iterations):
        proposals = rng.normal(mean, std, size=(population, *low.shape))
        proposals[0] = mean
        values = score(np.clip(proposals, low, high))
        order = np.argsort(values, kind="stable")[:elites]
        selected = proposals[order]
        mean = np.mean(selected, axis=0)
        std = np.maximum(np.std(selected, axis=0), 1e-5)

    final = np.stack((np.clip(initial, low, high), np.clip(mean, low, high)))
    final_scores = score(final)
    # Initial wins ties, matching the reported strict-improvement rule.
    winner = 1 if final_scores[1] < final_scores[0] else 0
    return CEMResult(final[winner].copy(), float(final_scores[winner]),
                     population * iterations + 2, iterations + 1)
