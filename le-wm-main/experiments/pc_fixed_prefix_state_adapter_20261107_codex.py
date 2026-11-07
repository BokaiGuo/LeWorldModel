"""Explicit state-assisted JEPA adapter for the shared fixed-prefix search."""
import numpy as np
import torch

from experiments.pc_fixed_prefix_shared_20261107_codex import search
from experiments.pc_input_permissions import STATE, StateAssistedPointMaze


def compile_state(boundary, current, goal, parents, probes, tails, *, mean, std,
                  objective_factory, prefix_steps, retain_count, strategy, budget, seed):
    if not isinstance(boundary, StateAssistedPointMaze) or boundary.audit.track_id != STATE:
        raise ValueError("state-assisted PointMaze boundary required")
    if set(current) != {"visual", "proprio"} or set(goal) != {"visual", "proprio"}:
        raise ValueError("explicit current and goal visual plus proprio required")
    for value in (current, goal):
        if value["visual"].ndim != 5 or value["visual"].shape[:2] != (1, 1) or value["proprio"].shape != (1, 1, 4):
            raise ValueError("single-frame visual and four-field proprio required")
    # Record goal permission as an actual boundary receipt, not only metadata.
    boundary._record("encode_goal", goal)
    target = boundary._model.encode(dict(goal))
    objective = objective_factory(target)
    mean, std = torch.as_tensor(mean), torch.as_tensor(std)
    if mean.shape != (2,) or std.shape != (2,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError("invalid official action statistics")
    mean, std = mean.to(current["visual"].device), std.to(current["visual"].device)
    calls = 0

    def score(chains):
        nonlocal calls
        if chains.shape[-1] != 2:
            raise ValueError("PointMaze requires 2D actions")
        values = []
        for chain in chains:
            raw = torch.as_tensor(chain, dtype=torch.float32, device=mean.device)
            packet = ((raw - mean) / std).reshape(-1, 1, 10)
            prediction = boundary.predict(current, packet)
            cost = objective(prediction, packet)
            if cost.numel() != 1 or not torch.isfinite(cost).all():
                raise ValueError("invalid JEPA objective output")
            values.append(float(cost.item()))
            calls += 1
        return np.asarray(values)

    result = search(score, parents, probes, tails, prefix_steps=prefix_steps, retain_count=retain_count,
                    strategy=strategy, budget=budget, seed=seed)
    result.update({"track": STATE, "predict_calls": calls, "goal_encoding_calls": 1})
    return result
