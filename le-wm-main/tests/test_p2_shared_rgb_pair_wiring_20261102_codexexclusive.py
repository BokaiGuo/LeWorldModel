from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from experiments.spectral_reachability import p2_shared_rgb_pair_wiring_20261102_codexexclusive as wire


@contextmanager
def scope(_meter):
    yield


class Adapter:
    def __init__(self, meter):
        self.m = meter

    def packet(self, fn):
        def nested():
            for k in ("action_encoder", "predict", "predictor", "pred_proj"):
                self.m.call(k, lambda: None)
            return fn()
        return self.m.call("packet", nested)

    def root_jacobian(self, context):
        return self.m.call("jacobian", lambda: (self.packet(lambda: None), self.packet(lambda: None),
                           np.vstack((np.diag(np.arange(20, 0, -1)), np.zeros((172, 20)))))[-1])

    def full_jacobian_at(self, context, anchor):
        return self.root_jacobian(context)

    def packet_jacobian(self, context):
        return self.m.call("jacobian", lambda: (self.packet(lambda: None),
                           np.vstack((np.diag(np.arange(10, 0, -1)), np.zeros((182, 10)))))[-1])

    def step(self, context, packet):
        def f():
            value = deepcopy(context)
            value["latent_history"][-1][0] += float(packet.sum()) / 100
            return value
        return self.packet(f)


def inputs():
    rgb = [np.full((224, 224, 3), i, np.uint8) for i in (5, 10, 15)]
    goal = np.zeros((224, 224, 3), np.uint8)
    actions = np.arange(30, dtype=np.float32).reshape(15, 2) / 100
    return rgb, goal, actions


def test_pair_uses_four_shared_encodings_and_actual_last_ten_actions():
    rgb, goal, actions = inputs()
    calls, events = [], []
    def encode(frame):
        calls.append(int(frame[0, 0, 0]))
        return np.full(192, calls[-1], np.float32)
    result = wire.run_pair(rgb, goal, actions, np.zeros(2), np.ones(2), encode,
                           Adapter, scope, events.append, base=0)
    assert calls == [5, 10, 15, 0]
    assert result["context"]["action_history"] == actions[-10:].reshape(2, 10).tolist()
    assert [x["arm"] for x in result["branches"]] == list(wire.kernel.ARMS)
    assert all(x["resource_counts"]["attempted"] == wire.kernel.CAPS for x in result["branches"])
    assert result["shared_setup"]["encode_returned"] == 4
    assert result["equal_total_compute"] is False


def test_encoder_failure_aborts_both_arms_and_is_logged():
    rgb, goal, actions = inputs()
    events = []
    def bad(_):
        raise RuntimeError("broken encoder")
    with pytest.raises(RuntimeError, match="broken"):
        wire.run_pair(rgb, goal, actions, np.zeros(2), np.ones(2), bad,
                      Adapter, scope, events.append, base=0)
    assert any(e["kind"] == "shared_encode_failure" for e in events)
    assert not any(e["kind"] == "branch" for e in events)


def test_action_domain_and_image_shape_rejected_before_encoding():
    rgb, goal, actions = inputs()
    actions[0, 0] = 2
    with pytest.raises(ValueError):
        wire.run_pair(rgb, goal, actions, np.zeros(2), np.ones(2), lambda _: pytest.fail(),
                      Adapter, scope, lambda _: None, base=0)


def test_capture_bundle_checks_images_receipts_and_sha(tmp_path):
    rgb, goal, actions = inputs()
    images = []
    for i, frame in enumerate([*rgb, goal]):
        path = tmp_path / f"frame{i}.png"
        Image.fromarray(frame).save(path)
        images.append({"file": path.name, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "pixel_sha256": wire.pixel_sha(frame)})
    rows = [{"step": i + 1, "dispatch": a.tolist(), "returned": a.tolist(),
             "terminated": False, "truncated": False} for i, a in enumerate(actions)]
    bundle = {"schema": wire.NAME + ".capture.v1", "base": 0,
              "frames": [dict(x, step=s) for x, s in zip(images[:3], (5, 10, 15))],
              "goal": images[3], "actions": rows}
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(bundle))
    digest = wire.checkpoint.sha(path)
    _, frames, observed_goal, observed_actions = wire.load_capture(path, digest)
    assert np.array_equal(frames[-1], rgb[-1])
    assert np.array_equal(observed_goal, goal)
    assert np.array_equal(observed_actions, actions)
    bundle["actions"][2]["returned"][0] += .001
    path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="actual action"):
        wire.load_capture(path, wire.checkpoint.sha(path))


def test_second_arm_failure_does_not_publish_pair_complete():
    rgb, goal, actions = inputs()
    events = []
    class FailingAdapter(Adapter):
        count = 0
        def __init__(self, meter):
            super().__init__(meter)
            FailingAdapter.count += 1
            if FailingAdapter.count == 2:
                raise RuntimeError("second arm")
    with pytest.raises(RuntimeError, match="second arm"):
        wire.run_pair(rgb, goal, actions, np.zeros(2), np.ones(2),
                      lambda _: np.zeros(192), FailingAdapter, scope, events.append, base=0)
    assert events[-1]["kind"] == "pair_failure"
    assert not any(e["kind"] == "pair_complete" for e in events)
