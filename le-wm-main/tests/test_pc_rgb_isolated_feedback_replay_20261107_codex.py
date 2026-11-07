import copy

import numpy as np
import pytest

from experiments.pc_rgb_isolated_feedback_replay_20261107_codex import (
    encode_image, validate_request,
)


def request():
    image = encode_image(np.zeros((224, 224, 3), dtype=np.uint8))
    return {"episode": 9000, "iteration": 0, "arm": "effect", "remaining": 25,
            "current": image, "goal": image, "old_tail": None}


def test_valid_request():
    current, goal, tail = validate_request(request())
    assert current.shape == goal.shape == (224, 224, 3)
    assert tail is None


@pytest.mark.parametrize("change", [
    {"state": [0, 0]}, {"success": True}, {"distance": 0}, {"remaining": 20},
    {"episode": 9001}, {"arm": "oracle"}, {"current": "AA=="},
    {"old_tail": [[0, 0]]},
])
def test_reject_privileged_or_invalid_fields(change):
    data = copy.deepcopy(request())
    data.update(change)
    with pytest.raises((ValueError, TypeError)):
        validate_request(data)


def test_tail_bounds_and_clock():
    data = request()
    data.update(iteration=1, remaining=20, old_tail=np.zeros((20, 2)).tolist())
    validate_request(data)
    data["old_tail"][0][0] = 2
    with pytest.raises(ValueError):
        validate_request(data)
