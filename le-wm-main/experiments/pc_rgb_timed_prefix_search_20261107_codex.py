"""Deadline-aware RGB prefix search with explicit one-packet fallback."""
import numpy as np
import torch

from experiments import pc_rgb_deadline_scoring_20261107_codex as deadline
from experiments.pc_input_permissions import RGB, RGBLeWM
from experiments.pc_scored_candidate_selection_20261107_codex import ScoredCandidate


def build(seed, old_tail, prefix_steps, count=3):
    if prefix_steps not in (5, 10) or type(seed) is not int or count < 1:
        raise ValueError("invalid prefix or seed")
    rng = np.random.default_rng(seed)
    parents = np.repeat(rng.uniform(-1, 1, (count, 5, 2)).astype(np.float32), 5, axis=1)
    if old_tail is not None:
        old = np.asarray(old_tail, dtype=np.float32)
        if old.ndim != 2 or old.shape[1] != 2 or len(old) > 25 or not np.isfinite(old).all() or np.abs(old).max() > 1:
            raise ValueError("invalid old tail")
        parents[0] = np.pad(old, ((0, 25-len(old)), (0, 0)))
    blocks = (25-prefix_steps)//5
    probes = np.repeat(rng.uniform(-1, 1, (2, blocks, 2)).astype(np.float32), 5, axis=1)
    tails = np.repeat(rng.uniform(-1, 1, (2, blocks, 2)).astype(np.float32), 5, axis=1)
    return parents, probes, tails


def compile_timed(boundary, current, *, scaler, old_tail, candidate_seed, strategy_seed, strategy, remaining_steps):
    if not isinstance(boundary, RGBLeWM) or boundary.audit.track_id != RGB:
        raise ValueError("audited RGB boundary required")
    if set(current) != {"pixels", "goal", "action"} or remaining_steps not in (5, 10, 15, 20, 25):
        raise ValueError("invalid observation or remaining clock")
    if strategy not in ("parent", "lookahead", "vector", "shuffled"):
        raise ValueError("unknown strategy")
    if remaining_steps > 5:
        prefix = min(10, remaining_steps - 5)
        parents, probes, tails = build(candidate_seed, old_tail, prefix)
        result = deadline.compile_rgb(boundary, current, parents, probes, tails, scaler=scaler,
                                      prefix_steps=prefix, retain_count=2, strategy=strategy,
                                      budget=13, seed=strategy_seed, remaining_steps=remaining_steps)
        result.update({"mode": "program", "prefix_steps": prefix})
        return result

    parents, _, _ = build(candidate_seed, old_tail, 5, count=13)
    if old_tail is None:
        parents[0] = 0
    normal = np.asarray(scaler.transform(parents[:, :5].reshape(-1, 2)), dtype=np.float32).reshape(1, 13, 1, 10)
    if not np.isfinite(normal).all():
        raise ValueError("invalid normalized candidates")
    scores, _ = boundary.score(current, torch.as_tensor(normal, device=current["pixels"].device))
    costs = scores.detach().cpu().numpy().reshape(-1)
    if costs.shape != (13,) or not np.isfinite(costs).all():
        raise ValueError("invalid fallback scores")
    index = int(np.argmin(costs))
    return {"track": RGB, "mode": "direct_fallback", "prefix_steps": None,
            "remaining_steps": remaining_steps, "model_packet_steps": 13, "forward_calls": 1,
            "model_queries": 13, "actions": parents[index].copy(),
            "choice": ScoredCandidate("fallback", index, float(costs[index])),
            "retained_prefixes": [], "parent_cost": costs, "probe_cost": np.empty((0, 0)),
            "tail_cost": np.empty(0)}
