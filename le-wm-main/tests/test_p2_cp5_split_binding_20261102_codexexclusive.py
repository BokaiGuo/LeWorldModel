import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability import p2_cp5_split_binding_20261102_codexexclusive as split


class FakeSession:
    def __init__(self, owner):
        self.owner = owner
        self.steps = 0

    def reconstruct(self):
        return {"root_sha256": "a" * 64, "task_id": "task",
                "rgb": np.zeros((2, 224, 224, 3), dtype=np.uint8),
                "history": np.zeros((2, 10), dtype="<f4"), "guard": "guard"}

    def step(self, action):
        self.steps += 1
        self.owner.actions.append(action.copy())
        return {"rgb": np.full((224, 224, 3), self.steps + self.owner.drift, dtype=np.uint8),
                "snapshot": {"step": self.steps}, "terminated": False,
                "truncated": self.owner.stop == self.steps}

    def close(self):
        self.owner.closed += 1


class FakeFactory:
    def __init__(self):
        self.actions = []
        self.closed = 0
        self.drift = 0
        self.stop = None

    def __call__(self):
        return FakeSession(self)


def setup(tmp_path):
    factory = FakeFactory()
    adapter = split.Adapter(tmp_path / "session", factory, "a" * 64, "task",
                            np.zeros((2, 10), dtype="<f4"),
                            [hashlib.sha256(bytes(224 * 224 * 3)).hexdigest()] * 2)
    return adapter, factory


def test_capture_and_replay_full_chain(tmp_path):
    adapter, factory = setup(tmp_path)
    actions = np.zeros((10, 2), dtype="<f4")
    capture = adapter.capture("conditional", 0, actions[:5])
    winner = {"candidate_index": 0, "raw_float32_hex": actions.tobytes().hex(),
              "feedback": {"capture_receipt_sha256": capture["capture_receipt_sha256"]}}
    adapter.replay_winner("conditional", winner)
    assert len(factory.actions) == 15
    assert factory.closed == 2
    assert (adapter.path / "replay_conditional.json").exists()


def test_replay_pixel_drift_blocks_suffix(tmp_path):
    adapter, factory = setup(tmp_path)
    actions = np.zeros((10, 2), dtype="<f4")
    c = adapter.capture("conditional", 0, actions[:5])
    factory.drift = 1
    with pytest.raises(ValueError, match="replay mismatch"):
        adapter.replay_winner("conditional", {"candidate_index": 0,
            "raw_float32_hex": actions.tobytes().hex(),
            "feedback": {"capture_receipt_sha256": c["capture_receipt_sha256"]}})
    assert len(factory.actions) == 10


def test_stop_does_not_pad_or_return_capture(tmp_path):
    adapter, factory = setup(tmp_path)
    factory.stop = 3
    with pytest.raises(RuntimeError, match="native stop"):
        adapter.capture("conditional", 0, np.zeros((5, 2), dtype="<f4"))
    assert len(factory.actions) == 3
    assert factory.closed == 1
    assert (adapter.path / "capture_conditional_00_failure.json").exists()


def test_wrong_prefix_blocks_replay_before_new_session(tmp_path):
    adapter, factory = setup(tmp_path)
    plan = np.zeros((10, 2), dtype="<f4")
    c = adapter.capture("conditional", 0, plan[:5])
    plan[0, 0] = 0.5
    with pytest.raises(ValueError, match="winner prefix"):
        adapter.replay_winner("conditional", {"candidate_index": 0,
            "raw_float32_hex": plan.tobytes().hex(),
            "feedback": {"capture_receipt_sha256": c["capture_receipt_sha256"]}})
    assert len(factory.actions) == 5


def test_existing_output_rejected(tmp_path):
    adapter, factory = setup(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        split.Adapter(adapter.path, factory, "a" * 64, "task",
                      np.zeros((2, 10), dtype="<f4"), ["b" * 64] * 2)
