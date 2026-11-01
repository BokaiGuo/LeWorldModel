import inspect

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_rgb_feedback_candidate_20261030_v1 as candidate


def frame(pusher=(40, 40), obj=(100, 80)):
    a = np.full((224, 224, 3), 255, dtype=np.uint8)
    y, x = pusher
    a[y:y+10, x:x+10] = candidate.rgb.BLUE
    y, x = obj
    a[y:y+20, x:x+10] = candidate.rgb.BLOCK
    return a


def test_causal_exact_interface_and_corrective_packet():
    assert list(inspect.signature(candidate.decide).parameters) == ["current_rgb", "goal_rgb", "model_first_packet"]
    model = np.full((5, 2), .4, dtype="<f4")
    output = candidate.decide(frame(), frame((40, 60)), model)
    assert output["decision"] == "rgb_pusher_correction"
    assert np.allclose(output["packet"][0], [.2, 0])
    assert np.all(output["packet"][1:] == 0)
    assert np.all(model == np.float32(.4))


def test_model_fallback_only_when_rgb_is_valid():
    model = np.full((5, 2), .4, dtype="<f4")
    assert candidate.decide(frame(), frame(), model)["decision"] == "model_packet"
    assert candidate.decide(frame(), frame(obj=(130, 80)), model)["decision"] == "model_packet"
    invalid = frame()
    invalid[40:50, 40:50] = 255
    result = candidate.decide(invalid, frame(), model)
    assert result["decision"] == "rgb_rejected" and result["packet"] is None


def test_raw_action_fails_closed_and_extra_truth_cannot_enter():
    image = frame()
    with pytest.raises(ValueError):
        candidate.decide(image, image, np.full((5, 2), 2, dtype="<f4"))
    with pytest.raises(TypeError):
        candidate.decide(image, image, np.zeros((5, 2), "<f4"), state=[1, 2])
