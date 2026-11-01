import numpy as np
import pytest

from experiments.spectral_reachability import p1_dev303_feedback_history_20261102_codex_exclusive as subject


def frame(i):
    return np.full((224, 224, 3), i, np.uint8)


def test_real_receipt_order_and_sliding_window():
    history = subject.History()
    history.reset(frame(0), 0)
    for i in range(1, 16):
        action = np.array([0.25, 0.], dtype="<f4")
        history.dispatched(action, step=i)
        history.returned(action, frame(i), step=i)
    context = history.context()
    assert context["frame_ids"] == list(range(5, 16))
    assert [int(x[0, 0, 0]) for x in context["frames"]] == list(range(5, 16))
    assert context["actions"].shape == (10, 2)
    assert np.all(context["actions"][:, 0] == 0.25)
    context["frames"][0][:] = 0
    assert history.context()["frames"][0][0, 0, 0] == 5


@pytest.mark.parametrize("bad", ["wrong_step", "wrong_action", "missing_dispatch", "missing_frame"])
def test_bad_receipts_latch_failure(bad):
    history = subject.History()
    history.reset(frame(0), 0)
    if bad != "missing_dispatch":
        history.dispatched([0., 0.], step=1)
    with pytest.raises((ValueError, TypeError)):
        history.returned([1., 0.] if bad == "wrong_action" else [0., 0.],
                         None if bad == "missing_frame" else frame(1),
                         step=2 if bad == "wrong_step" else 1)
    with pytest.raises(ValueError):
        history.dispatched([0., 0.], step=1)


def test_incomplete_history_cannot_supply_context():
    history = subject.History()
    history.reset(frame(0), 0)
    with pytest.raises(ValueError):
        history.context()
