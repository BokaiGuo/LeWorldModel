import numpy as np
import pytest

from experiments.pc_rgb_spawn_planner_20261107_codex import validate_request


def request():
    return {"pixels": np.zeros((224, 224, 3), dtype=np.uint8),
            "goal": np.zeros((224, 224, 3), dtype=np.uint8), "old_tail": None,
            "remaining_steps": 25, "candidate_seed": 1, "strategy_seed": 2, "strategy": "lookahead"}


def test_privileged_and_unexpected_fields_fail_closed():
    for name in ("state", "proprio", "contact", "future_success"):
        message = request()
        message[name] = np.zeros(4)
        with pytest.raises(ValueError):
            validate_request(message)


def test_clock_and_actions_are_validated():
    message = request()
    message["remaining_steps"] = 20
    message["old_tail"] = np.zeros((20, 2), dtype=np.float32)
    validate_request(message)
    message["old_tail"][0, 0] = 1.1
    with pytest.raises(ValueError):
        validate_request(message)
    message = request()
    message["pixels"] = np.zeros((224, 224, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        validate_request(message)
