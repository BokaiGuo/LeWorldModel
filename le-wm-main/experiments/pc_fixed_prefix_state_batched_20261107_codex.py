"""Optional batched state-assisted adapter; original serial adapter is untouched."""
import numpy as np
import torch

from experiments.pc_fixed_prefix_shared_20261107_codex import search
from experiments.pc_input_permissions import STATE, StateAssistedPointMaze


def compile_state_batched(boundary, current, goal, parents, probes, tails, *, mean, std,
                          objective_factory, prefix_steps, retain_count, strategy,
                          budget, seed, batch_size):
    if not isinstance(boundary, StateAssistedPointMaze) or boundary.audit.track_id != STATE:
        raise ValueError("state-assisted boundary required")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive batch size required")
    if set(current) != {"visual", "proprio"} or set(goal) != {"visual", "proprio"}:
        raise ValueError("current and goal visual plus proprio required")
    for value in (current, goal):
        if value["visual"].ndim != 5 or value["visual"].shape[:2] != (1, 1) or value["proprio"].shape != (1, 1, 4):
            raise ValueError("single-frame visual and four-field proprio required")
    mean, std = torch.as_tensor(mean), torch.as_tensor(std)
    if (mean.shape != (2,) or std.shape != (2,) or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or torch.any(std <= 0)):
        raise ValueError("invalid action statistics")
    mean, std = mean.to(current["visual"].device), std.to(current["visual"].device)
    boundary._record("encode_goal", goal)
    target = boundary._model.encode(dict(goal))
    objective = objective_factory(target)
    calls = 0
    batch_lengths = []

    def score(chains):
        nonlocal calls
        if chains.shape[-1] != 2:
            raise ValueError("PointMaze requires 2D actions")
        values = []
        for start in range(0, len(chains), batch_size):
            raw = torch.as_tensor(chains[start:start + batch_size], dtype=torch.float32, device=mean.device)
            count, horizon, _ = raw.shape
            # JEPA convention is (model_time, candidate_batch, packet_width).
            packets = ((raw - mean) / std).reshape(count, horizon // 5, 10).transpose(0, 1).contiguous()
            expanded = {key: val.expand(count, *val.shape[1:]).contiguous() for key, val in current.items()}
            prediction = boundary.predict(expanded, packets)
            cost = objective(prediction, packets)
            if cost.shape != (count,) or not torch.isfinite(cost).all():
                raise ValueError("invalid batched JEPA objective output")
            values.extend(cost.detach().cpu().tolist())
            calls += 1
            batch_lengths.append(count)
        return np.asarray(values)

    result = search(score, parents, probes, tails, prefix_steps=prefix_steps, retain_count=retain_count,
                    strategy=strategy, budget=budget, seed=seed)
    result.update({"track": STATE, "predict_calls": calls, "batch_lengths": batch_lengths,
                   "goal_encoding_calls": 1})
    return result
