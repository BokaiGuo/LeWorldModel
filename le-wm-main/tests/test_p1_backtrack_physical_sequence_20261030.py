import numpy as np
import pytest

from experiments.spectral_reachability.p1_backtrack_physical_sequence_20261030 import sequence


def test_all_actions_and_four_checkpoints_with_guard_before_six():
    events = []
    raw = np.zeros((15, 2), dtype=np.float32)

    def dispatch(step, action):
        events.append(("dispatch", step, action.tobytes()))
        return dict(terminated=False, truncated=False)

    def capture(step, _transition):
        events.append(("capture", step))
        return {"step": step}

    def guard(step, observation):
        assert observation == {"step": 5}
        events.append(("guard", step))
        return True

    result = sequence(raw, dispatch, capture, guard)
    assert result["status"] == "horizon_completed"
    assert list(result["checkpoints"]) == [0, 5, 10, 15]
    assert result["checkpoints"][15] == {"step": 15}
    assert events.index(("guard", 5)) < next(i for i, e in enumerate(events) if e[:2] == ("dispatch", 6))


def test_guard_failure_and_early_stop_do_not_dispatch_suffix():
    calls = []
    def dispatch(step, action):
        calls.append(step)
        return dict(terminated=False, truncated=False)
    with pytest.raises(ValueError, match="prefix guard"):
        sequence(np.zeros((15, 2), np.float32), dispatch, lambda step, _: step, lambda *_: False)
    assert calls == [1, 2, 3, 4, 5]
    calls.clear()
    def early(step, action):
        calls.append(step)
        return dict(terminated=step == 3, truncated=False)
    out = sequence(np.zeros((15, 2), np.float32), early, lambda step, _: step, lambda *_: True)
    assert calls == [1, 2, 3]
    assert out["checkpoints"][5] is out["checkpoints"][10] is out["checkpoints"][15] is None


def test_rejects_action_rounding_and_missing_termination():
    with pytest.raises(ValueError, match="literal"):
        sequence(np.full((15, 2), .1), lambda *_: {}, lambda *_: 1, lambda *_: True)
    with pytest.raises(ValueError, match="termination"):
        sequence(np.zeros((15, 2), np.float32), lambda *_: {}, lambda *_: 1, lambda *_: True)
