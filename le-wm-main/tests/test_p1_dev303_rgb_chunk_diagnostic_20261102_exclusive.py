import hashlib

import numpy as np
import pytest
from PIL import Image

from experiments.spectral_reachability import (
    p1_dev303_rgb_chunk_diagnostic_20261102_exclusive as subject,
)


def rgb_image():
    image = np.zeros((224, 224, 3), np.uint8)
    image[98:103, 18:23] = subject.control.rgb.BLUE
    image[98:103, 80:85] = subject.control.rgb.BLOCK
    return image


class Store:
    def __init__(self, path):
        self.path = path
        self.events = []

    def emit(self, kind, **fields):
        self.events.append({"kind": kind, **fields})

    def check(self):
        pass


class FakeEnv:
    def __init__(self, *, mismatch_at=None, stop_at=None):
        self.count = 0
        self.mismatch_at = mismatch_at
        self.stop_at = stop_at
        self.latest_action = None
        self.goal = np.zeros((224, 224, 3), np.uint8)
        self.state = np.zeros(5)

    def reset(self, seed):
        assert seed == subject.SEED
        return {"state": self.state.copy()}, {"goal_state": self.state.copy(), "goal": self.goal}

    def step(self, action):
        self.count += 1
        self.latest_action = action.copy()
        if self.count == self.mismatch_at:
            self.latest_action[0] += 0.1
        return ({"state": self.state.copy()}, 0.0, self.count == self.stop_at, False,
                {"goal_state": self.state.copy(), "goal": self.goal, "n_contacts": 0})


def capture(store, env, arm, phase, step, expected=None):
    name = f"{arm}_{phase}_{step:02d}.png"
    Image.fromarray(rgb_image()).save(store.path / name)
    value = {"file": name, "pixel_sha256": hashlib.sha256(rgb_image().tobytes()).hexdigest()}
    store.emit("observation", arm=arm, **value)
    return value


def fixture():
    state = [0.0] * 5
    return ({"goal": state, "final_state": state,
             "frames": [{"step": i} for i in (0, 5, 10)]},
            {"state": state}, np.zeros((224, 224, 3), np.uint8))


@pytest.mark.parametrize("arm", subject.ARMS)
def test_each_arm_records_all_warmup_and_fixed_control_steps(tmp_path, arm):
    history, reset, goal = fixture()
    store, env = Store(tmp_path), FakeEnv()
    row = subject.run_branch(env, store, arm, history, reset, goal, capture=capture)
    assert row["warmup_returned"] == 10
    assert row["control_returned"] == 5
    assert row["status"] == "fixed_horizon_completed"
    assert len(row["observations"]) == 8
    assert len([event for event in store.events if event["kind"] == "action_observed"]) == 15
    if arm == "zero_prefix":
        assert all(not any(event["action"]) for event in row["transitions"])
    else:
        decisions = [event for event in store.events if event["kind"] == "controller_decision"]
        assert [event["proposal"]["frame_id"] for event in decisions] == list(range(5))
        assert all(event["proposal"]["P1_admission"] is False for event in decisions)


def test_action_mismatch_is_recorded_before_failure(tmp_path):
    history, reset, goal = fixture()
    store, env = Store(tmp_path), FakeEnv(mismatch_at=11)
    row = {"arm": "rgb_chunk", "status": "running", "warmup_returned": 0,
           "control_returned": 0, "transitions": [], "observations": []}
    with pytest.raises(ValueError, match="actual action mismatch"):
        subject.run_branch(env, store, "rgb_chunk", history, reset, goal, capture=capture, row=row)
    assert row["warmup_returned"] == 10
    assert store.events[-1]["kind"] == "action_observed"
    assert store.events[-1]["phase"] == "control"


def test_early_stop_records_no_padding(tmp_path):
    history, reset, goal = fixture()
    store, env = Store(tmp_path), FakeEnv(stop_at=12)
    row = subject.run_branch(env, store, "zero_prefix", history, reset, goal, capture=capture)
    assert row["status"] == "native_early_stop"
    assert row["control_returned"] == 2
    assert len(row["transitions"]) == 2


def test_sealed_controller_is_current_single_component_version():
    assert subject.base.sha(subject.control.__file__) == subject.CONTROL_SHA
    assert 'obj["components"] == 1' in open(subject.control.__file__, encoding="utf-8").read()


def test_contract_is_development_only_and_binds_controller():
    contract = subject.build()
    assert contract["arms_in_fixed_order"] == list(subject.ARMS)
    assert contract["rgb_controller_sha256"] == subject.CONTROL_SHA
    assert contract["bindings"][str(subject.Path(subject.control.__file__).resolve())] == subject.CONTROL_SHA
    assert contract["selection_uses_state_goal_reward_contact_or_outcome"] is False
    assert contract["P1_admission"] is contract["independent_confirmation"] is False


def test_exclusive_write(tmp_path):
    path = tmp_path / "contract.json"
    subject.write_new(path, {"P1_admission": False})
    with pytest.raises(FileExistsError):
        subject.write_new(path, {"P1_admission": False})
