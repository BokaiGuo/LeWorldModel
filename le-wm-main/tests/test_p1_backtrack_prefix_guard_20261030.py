import pytest
import numpy as np

from experiments.spectral_reachability.p1_backtrack_prefix_guard_20261030 import (
    establish_or_verify, identity,
)
from experiments.spectral_reachability.p1_backtrack_physical_sequence_20261030 import sequence


def fields():
    return dict(task_id="a" * 64, condition="off", prefix_hex="12" * 40,
                state=[0.0] * 7, snapshot={"body": [1, 2]}, pixels={"off": "b" * 64, "on": "c" * 64},
                history_hex="34" * 80 + "12" * 40)


def test_create_replay_and_reject_observation_or_prefix_drift(tmp_path):
    path = tmp_path / "prefix.json"
    base = identity(**fields())
    assert establish_or_verify(path, base) == establish_or_verify(path, base)
    changed = fields()
    changed["snapshot"] = {"body": [1, 3]}
    with pytest.raises(ValueError, match="mismatch"):
        establish_or_verify(path, identity(**changed))
    assert path.read_bytes() == base
    changed = fields()
    changed["prefix_hex"] = "ff" * 40
    with pytest.raises(ValueError, match="history disagree"):
        identity(**changed)


def test_rejects_symlink_and_nonfinite(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(identity(**fields()))
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        establish_or_verify(link, target.read_bytes())
    changed = fields()
    changed["state"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        identity(**changed)


def test_replay_identity_mismatch_prevents_sixth_action(tmp_path):
    path = tmp_path / "reference"
    establish_or_verify(path, identity(**fields()))
    changed = fields()
    changed["pixels"] = {"off": "d" * 64, "on": "c" * 64}
    dispatched = []

    def dispatch(step, _action):
        dispatched.append(step)
        return {"terminated": False, "truncated": False}

    def guard(_step, _observation):
        establish_or_verify(path, identity(**changed))
        return True

    with pytest.raises(ValueError, match="mismatch"):
        sequence(np.zeros((15, 2), np.float32), dispatch, lambda step, _: step, guard)
    assert dispatched == [1, 2, 3, 4, 5]
