"""Development-only temporal near-tie resolution with endpoint secondary ranking."""
import numpy as np

from experiments import pc_rgb_temporal_final_search_20261107_codex as original
from experiments.pc_scored_candidate_selection_20261107_codex import ScoredCandidate

TOLERANCE = 1e-3


def choose_index(temporal, endpoint, tolerance=TOLERANCE):
    temporal, endpoint = np.asarray(temporal, np.float64), np.asarray(endpoint, np.float64)
    if (temporal.ndim != 1 or not len(temporal) or temporal.shape != endpoint.shape
            or not np.isfinite(temporal).all() or not np.isfinite(endpoint).all()
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("invalid scored pool")
    pool = np.flatnonzero(temporal <= temporal.min() + tolerance)
    secondary = endpoint[pool]
    return int(pool[np.flatnonzero(secondary <= secondary.min() + tolerance)[0]])


def compile_rgb(boundary, current, *, scaler, old_tail, candidate_seed, strategy_seed, strategy, remaining_steps):
    result = original.compile_rgb(boundary, current, scaler=scaler, old_tail=old_tail,
        candidate_seed=candidate_seed, strategy_seed=strategy_seed, strategy=strategy,
        remaining_steps=remaining_steps, final_rule="temporal")
    if remaining_steps == 5:
        return result
    temporal = np.concatenate([np.asarray(x) for x in result["temporal_scores"]])
    endpoint = np.concatenate((np.asarray(result["parent_cost"]).ravel(),
                               np.asarray(result["probe_cost"]).ravel(),
                               np.asarray(result["tail_cost"]).ravel()))
    if temporal.shape != (16,) or endpoint.shape != (16,):
        raise ValueError("unexpected candidate count")
    index = choose_index(temporal, endpoint)
    synthetic = [np.ones(len(group), dtype=np.float32) for group in result["temporal_scores"]]
    synthetic_flat = np.concatenate(synthetic)
    synthetic_flat[index] = 0
    synthetic = [synthetic_flat[:3], synthetic_flat[3:12], synthetic_flat[12:]]
    prefix = min(10, remaining_steps - 5)
    parents, probes, tails = original.base.timed.build(candidate_seed, old_tail, prefix)
    parents, probes, tails = (parents[:, :remaining_steps], probes[:, :remaining_steps-prefix],
                              tails[:, :remaining_steps-prefix])
    selected, actions = original.choose_with_temporal(
        result, synthetic, parents, probes, tails, prefix, remaining_steps)
    result["choice"] = ScoredCandidate(selected.source, selected.index, float(temporal[index]))
    result["actions"] = actions
    result["selected_endpoint_cost"] = float(endpoint[index])
    result["temporal_min"] = float(temporal.min())
    result["tiebreak_index"] = index
    return result
