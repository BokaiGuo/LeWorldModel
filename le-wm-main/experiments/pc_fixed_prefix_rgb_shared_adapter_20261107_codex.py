"""RGB LeWM adapter for the track-independent fixed-prefix search core."""
import numpy as np
import torch

from experiments.pc_fixed_prefix_shared_20261107_codex import search
from experiments.pc_input_permissions import RGB, RGBLeWM


def compile_rgb(boundary, current, parents, probes, tails, *, scaler, prefix_steps,
                retain_count, strategy, budget, seed):
    if not isinstance(boundary, RGBLeWM) or boundary.audit.track_id != RGB:
        raise ValueError("audited RGB-only LeWM boundary required")
    if set(current) != {"pixels", "goal", "action"}:
        raise ValueError("RGB, goal, and legal action history required")
    forward_calls = 0

    def score(chains):
        nonlocal forward_calls
        count, horizon, dim = chains.shape
        normal = np.asarray(scaler.transform(chains.reshape(-1, dim)), dtype=np.float32).reshape(chains.shape)
        if not np.isfinite(normal).all():
            raise ValueError("nonfinite normalized action")
        costs = []
        for start in range(0, count, 32):
            packet = torch.as_tensor(normal[start:start + 32].reshape(1, -1, horizon // 5, 5 * dim),
                                     dtype=torch.float32, device=current["pixels"].device)
            value, _ = boundary.score(current, packet)
            result = value.detach().cpu().numpy().reshape(-1)
            if len(result) != len(packet[0]) or not np.isfinite(result).all():
                raise ValueError("invalid model output")
            costs.append(result)
            forward_calls += 1
        return np.concatenate(costs)

    result = search(score, parents, probes, tails, prefix_steps=prefix_steps, retain_count=retain_count,
                    strategy=strategy, budget=budget, seed=seed)
    result.update({"track": RGB, "forward_calls": forward_calls})
    return result
