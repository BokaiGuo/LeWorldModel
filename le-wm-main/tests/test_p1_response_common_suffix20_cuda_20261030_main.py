import copy

import numpy as np
import pytest

from experiments.spectral_reachability import p1_response_common_suffix20_cuda_20261030_main as subject


def example():
    prefix = np.full((5, 2), 0.25, dtype="<f4")
    states = np.arange(3 * 192, dtype="<f4").reshape(3, 192)
    history = np.arange(20, dtype="<f4").reshape(2, 10)
    task = {"root_context": {"latent_history": states.tolist(), "action_history": history.tolist()},
            "action_mean": [0.0, 0.0], "action_std": [1.0, 1.0]}
    candidate = {"raw_float32_bits": prefix.tobytes().hex() + ("00" * 40),
                 "prefix": prefix.tolist(),
                 "mid": {"latent_history": np.vstack((states[1:], np.zeros((1, 192)))).tolist(),
                         "action_history": np.vstack((history[-1], prefix.reshape(10))).tolist()}}
    return task, candidate


def test_context_binding_and_tampering():
    task, candidate = example()
    subject.check_context(task, candidate)
    bad = copy.deepcopy(candidate)
    bad["prefix"][0][0] = -0.25
    with pytest.raises(ValueError, match="prefix action bits"):
        subject.check_context(task, bad)
    bad = copy.deepcopy(candidate)
    bad["mid"]["action_history"][1][0] = -0.25
    with pytest.raises(ValueError, match="context binding"):
        subject.check_context(task, bad)
