"""Use the inverse bounded parameterization for the original reference runner."""
from pathlib import Path

import torch

from experiments import pc_response_temporal_v3_reacher_adapter_preflight_20261104_codex as run

PREFIX = run.ROOT / "results/pc_response_temporal_v3_reacher_adapter_preflight_v2_20261104_codex"
PROTOCOL = run.ROOT.parent / "docs/PC_RESPONSE_TEMPORAL_V3_REACHER_ADAPTER_PREFLIGHT_V2_PROTOCOL_20261104_CODEX.md"
original_dependencies = run.dependencies
original_terminal = run.p0._terminal_batch


def inverse_bounded(normalized, lower, upper):
    midpoint, radius = (lower + upper) / 2, (upper - lower) / 2
    ratio = (normalized - midpoint) / radius
    if not torch.isfinite(ratio).all() or torch.any(torch.abs(ratio) >= 1):
        raise ValueError("inverse reference requires strictly interior actions")
    params = torch.atanh(ratio)
    torch.testing.assert_close(midpoint + radius * torch.tanh(params), normalized, rtol=1e-6, atol=1e-6)
    return params


def terminal_reference(model, pixels, goal, history, normalized, *, action_lower, action_upper):
    params = inverse_bounded(normalized, action_lower, action_upper)
    return original_terminal(model, pixels, goal, history, params,
                             action_lower=action_lower, action_upper=action_upper)


def dependencies(source):
    return {**original_dependencies(source), str(Path(__file__).resolve()): run.host.bridge.sha_file(Path(__file__))}


if __name__ == "__main__":
    run.PREFIX, run.PROTOCOL = PREFIX, PROTOCOL
    run.MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
    run.CONTEXT = PREFIX.with_name(PREFIX.name + "_context.npz")
    run.RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
    run.dependencies, run.p0._terminal_batch = dependencies, terminal_reference
    run.main()
