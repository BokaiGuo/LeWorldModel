import numpy as np
import pytest

from experiments.spectral_reachability import p1_dev303_rgb_feedback_scored_tail_20261102_codex_exclusive as subject


def image(pusher=(20, 100), object_x=80):
    value = np.zeros((224, 224, 3), np.uint8)
    x, y = pusher
    value[y-2:y+3, x-2:x+3] = subject.rgb.BLUE
    value[98:103, object_x:object_x+5] = subject.rgb.BLOCK
    return value


class SyntheticBackend(subject.CUDARealScorer):
    """CPU test double; not evidence of a model or CUDA run."""
    def __init__(self):
        self.calls = 0
        self.closed = False

    def begin(self, frames, actions):
        self.frames = [x.copy() for x in frames]
        self.actions = actions.copy()

    def score(self, variants):
        assert variants.shape == (64, 5, 2, 5, 2)
        self.calls += 1
        result = np.ones((64, 5))
        result[0] = 0
        return result

    def finish(self):
        assert self.calls == 5
        return {"synthetic_calls": self.calls}

    def close(self):
        self.closed = True


def feedback():
    return subject.Feedback([image() for _ in range(11)], np.zeros((10, 2), np.float32),
                            frame_ids=list(range(11)))


def test_feedback_reobserves_and_updates_actual_history():
    controller, backend = feedback(), SyntheticBackend()
    first = controller.plan(backend)
    assert backend.calls == 5 and first["model_scoring"]["candidate"] == 0
    assert first["action"] == first["approach_candidate"]
    assert backend.closed
    with pytest.raises(ValueError):
        controller.plan(SyntheticBackend())
    assert controller.closed


def test_next_step_uses_new_rgb_and_actual_action():
    controller = feedback()
    first = controller.plan(SyntheticBackend())
    controller.accept(image((23, 100)), frame_id=11, actual_action=first["action"])
    backend = SyntheticBackend()
    second = controller.plan(backend)
    assert second["frame_id"] == 11
    assert second["gap_pixels"] < first["gap_pixels"]
    assert np.array_equal(backend.actions[-1], np.asarray(first["action"], np.float32))
    assert np.array_equal(backend.frames[-1], image((23, 100)))
    assert second["actual_history_sha256"] != first["actual_history_sha256"]


@pytest.mark.parametrize("wrong_frame,wrong_action", [(10, False), (12, False), (11, True)])
def test_bad_receipt_latches_closed(wrong_frame, wrong_action):
    controller = feedback()
    first = controller.plan(SyntheticBackend())
    action = [0., 0.] if wrong_action else first["action"]
    with pytest.raises(ValueError):
        controller.accept(image(), frame_id=wrong_frame, actual_action=action)
    with pytest.raises(ValueError):
        controller.plan(SyntheticBackend())


def test_incomplete_history_rejected():
    with pytest.raises(ValueError):
        subject.Feedback([image()] * 10, np.zeros((10, 2)), frame_ids=list(range(10)))


def test_clipped_or_fragmented_rgb_fails_closed():
    for pixels in (image(object_x=0), image()):
        if tuple(pixels[100, 0]) != subject.rgb.BLOCK:
            pixels[120:123, 100:103] = subject.rgb.BLOCK
        control = subject.Feedback([pixels] * 11, np.zeros((10, 2)), frame_ids=list(range(11)))
        with pytest.raises(ValueError):
            control.plan(SyntheticBackend())
        assert control.closed


def test_near_object_abstains_before_scoring():
    pixels = image(pusher=(20, 100), object_x=27)
    controller = subject.Feedback([pixels] * 11, np.zeros((10, 2)), frame_ids=list(range(11)))
    backend = SyntheticBackend()
    assert controller.plan(backend)["status"] == "near_object_abstain_no_contact_claim"
    assert backend.calls == 0


def test_contract_and_budget():
    p = subject.protocol()
    assert p["budget_per_replan"]["predict_samples"] == 3200
    assert p["budget_per_replan"]["rgb_encode_calls"] == 3
    assert not p["physical_execution_authorized"] and not p["P1_admission"]
