"""Twenty-update exploratory continuation with explicit prefix/context checks."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_response_common_suffix_cuda_20261030_main as prior

NAME = "p1_response_common_suffix20_cuda_20261030_main"


def check_context(task, candidate):
    raw = bytes.fromhex(candidate["raw_float32_bits"])
    if len(raw) != 80:
        raise ValueError("full candidate action bytes")
    prefix = np.frombuffer(raw[:40], dtype="<f4").reshape(5, 2)
    if not np.array_equal(prefix, np.asarray(candidate["prefix"], dtype="<f4")):
        raise ValueError("prefix action bits")
    root = task["root_context"]
    mid = candidate["mid"]
    root_states = np.asarray(root["latent_history"], dtype="<f4")
    mid_states = np.asarray(mid["latent_history"], dtype="<f4")
    root_actions = np.asarray(root["action_history"], dtype="<f4")
    mid_actions = np.asarray(mid["action_history"], dtype="<f4")
    mean = np.asarray(task["action_mean"], dtype="<f4")
    std = np.asarray(task["action_std"], dtype="<f4")
    if (root_states.shape != (3, 192) or mid_states.shape != (3, 192)
            or root_actions.shape != (2, 10) or mid_actions.shape != (2, 10)
            or mean.shape != (2,) or std.shape != (2,) or not np.isfinite(std).all()
            or not (std > 0).all()):
        raise ValueError("context shapes or normalizer")
    expected = ((prefix - mean) / std).reshape(10)
    if (not np.array_equal(mid_states[:2], root_states[1:])
            or not np.array_equal(mid_actions[0], root_actions[-1])
            or not np.allclose(mid_actions[1], expected, atol=1e-6, rtol=1e-6)):
        raise ValueError("prefix to predicted context binding")


def configure():
    original_optimize = prior.optimize
    original_bindings = prior.base.bindings

    def checked_optimize(model, task, candidate, device):
        check_context(task, candidate)
        return original_optimize(model, task, candidate, device)

    def bindings():
        result = original_bindings()
        path = Path(__file__).resolve()
        result[str(path)] = prior.base.sha(path)
        return result

    prior.optimize = checked_optimize
    prior.base.bindings = bindings
    prior.STEPS = 20
    prior.NAME = NAME


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configure()
    print("Run 20-update model-only comparison after prefix/context checks on host CUDA.", flush=True)
    prior.run(args.output)
