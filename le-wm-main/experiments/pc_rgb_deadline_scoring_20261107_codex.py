"""RGB-only shared fixed-prefix scoring at an explicit remaining deadline."""
import numpy as np
import torch

from experiments.pc_fixed_prefix_shared_20261107_codex import search
from experiments.pc_input_permissions import RGB, RGBLeWM


def compile_rgb(boundary, current, parents, probes, tails, *, scaler, prefix_steps,
                retain_count, strategy, budget, seed, remaining_steps):
    if not isinstance(boundary, RGBLeWM) or boundary.audit.track_id != RGB:
        raise ValueError("audited RGB boundary required")
    if set(current) != {"pixels", "goal", "action"}:
        raise ValueError("invalid RGB observation")
    if type(remaining_steps) is not int or remaining_steps not in (5, 10, 15, 20, 25):
        raise ValueError("deadline must match positive five-step packets")
    forward_calls, model_packet_steps = 0, 0

    def score(chains):
        nonlocal forward_calls, model_packet_steps
        count, horizon, dim = chains.shape
        if horizon != 25:
            raise ValueError("expected declared 25-step action horizon")
        raw = chains[:, :remaining_steps]
        normalized = np.asarray(scaler.transform(raw.reshape(-1, dim)), dtype=np.float32)
        if not np.isfinite(normalized).all():
            raise ValueError("nonfinite normalized action")
        normalized = normalized.reshape(count, remaining_steps // 5, 5 * dim)
        costs = []
        for start in range(0, count, 32):
            chunk = torch.as_tensor(normalized[start:start+32][None],
                                    dtype=torch.float32, device=current["pixels"].device)
            value, _ = boundary.score(current, chunk)
            result = value.detach().cpu().numpy().reshape(-1)
            if len(result) != len(chunk[0]) or not np.isfinite(result).all():
                raise ValueError("invalid model output")
            costs.append(result)
            forward_calls += 1
            model_packet_steps += len(result) * (remaining_steps // 5)
        return np.concatenate(costs)

    result = search(score, parents, probes, tails, prefix_steps=prefix_steps, retain_count=retain_count,
                    strategy=strategy, budget=budget, seed=seed)
    result.update({"track": RGB, "forward_calls": forward_calls,
                   "model_packet_steps": model_packet_steps, "remaining_steps": remaining_steps})
    return result
