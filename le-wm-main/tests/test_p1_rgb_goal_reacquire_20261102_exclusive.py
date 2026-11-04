import numpy as np
import pytest

from experiments.spectral_reachability import p1_rgb_goal_reacquire_20261102_exclusive as gate
from experiments.spectral_reachability import p1_dev305_goal_reacquire_contrast_20261102_exclusive as contrast


def pixels():
    image = np.full((224, 224, 3), 255, np.uint8)
    image[50:70, 50:70] = [120, 140, 160]
    image[100:110, 100:110] = [50, 100, 230]
    return image


def packet(n):
    return {"pixels": pixels(), "capture_id": str(n), "sequence": n, "captured_ns": n,
            "session_id": "s", "camera_id": "c", "target_id": "t"}


def invoke(goals, current, selector=lambda _g, _c: [0.1, -0.2]):
    return gate.choose(goals, current, now_ns=3, max_age_ns=3, selector=selector)


def test_valid_wiring_and_no_source_mutation():
    first, second, current = packet(1), packet(2), packet(3)
    before = first["pixels"].copy()

    def selector(goal, _current):
        goal[:] = 0
        return [0.1, -0.2]

    result = invoke([first, second], current, selector)
    assert result["status"] == "ready" and result["action"] == [0.1, -0.2]
    assert np.array_equal(first["pixels"], before)


@pytest.mark.parametrize("kind", ["missing_second", "invalid_goal", "duplicate", "context",
                                  "stale", "masks", "current", "malformed", "guard"])
def test_rejection_never_calls_selector(kind, monkeypatch):
    a, b, c = packet(1), packet(2), packet(3)
    goals = [a, b]
    if kind == "missing_second":
        goals = [a]
    elif kind == "invalid_goal":
        a["pixels"][50:70, 50:70] = 255
        a["pixels"][210:224, 50:70] = [120, 140, 160]
    elif kind == "duplicate":
        b["capture_id"] = a["capture_id"]
    elif kind == "context":
        b["target_id"] = "other"
    elif kind == "stale":
        a["captured_ns"] = 0
        a["sequence"] = 0
        b["captured_ns"] = 1
        b["sequence"] = 1
        c["captured_ns"] = 2
        c["sequence"] = 2
    elif kind == "masks":
        b["pixels"][50, 50] = 255
    elif kind == "current":
        c["pixels"][:] = 255
    elif kind == "malformed":
        a.pop("pixels")
    elif kind == "guard":
        monkeypatch.setattr(gate, "_guard_intact", lambda: False)
    calls = []
    result = gate.choose(goals, c, now_ns=10 if kind == "stale" else 3,
                         max_age_ns=3, selector=lambda *_: calls.append(1))
    assert result["status"] == "blocked" and result["action"] is None
    assert result["selector_called"] is False and not calls


@pytest.mark.parametrize("result", [[2, 0], [float("nan"), 0], [0], None])
def test_invalid_selector_output_blocks(result):
    decision = invoke([packet(1), packet(2)], packet(3), lambda *_: result)
    assert decision["status"] == "blocked" and decision["action"] is None
    assert decision["selector_called"]


def test_selector_error_is_not_scientific_failure():
    def broken(*_):
        raise RuntimeError("engineering failure")

    result = invoke([packet(1), packet(2)], packet(3), broken)
    assert result["status"] == "blocked" and result["action"] is None
    assert result["selector_called"] and result["details"]["exception_type"] == "RuntimeError"


def test_real_saved_root_blocks_before_selection():
    report = contrast.analyze()
    assert report["decision"]["reason"] == "goal_visibility_failed"
    assert not report["decision"]["selector_called"]
    assert report["executed_action"] is None
    assert report["reacquisition"]["available_real_goal_captures"] == 1
