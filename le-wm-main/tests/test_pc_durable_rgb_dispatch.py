import hashlib
import json

import numpy as np
import pytest

from experiments.pc_durable_rgb_dispatch import dispatch, inspect


class Environment:
    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def step(self, action):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("interrupted")
        return np.full((224, 224, 3), self.calls, np.uint8)


def test_dispatch_persists_each_action_and_refuses_overwrite(tmp_path):
    env = Environment()
    path = tmp_path / "step"
    frames, times = dispatch(env, [[.1, 0], [0, .2]], path)
    assert len(frames) == len(times) == 2
    assert json.loads((path / "000_attempt.json").read_text())["action"] == [.1, 0]
    assert np.array_equal(np.load(path / "001_rgb.npy"), frames[1])
    assert json.loads((path / "001_returned.json").read_text())["rgb_sha256"] == hashlib.sha256(frames[1].tobytes()).hexdigest()
    assert inspect(path) == {"attempts": 2, "returned": 2, "pending": None, "safe_to_retry": False}
    with pytest.raises(FileExistsError):
        dispatch(env, [[0, 0]], path)
    assert env.calls == 2


def test_interrupted_action_remains_ambiguous_and_not_retried(tmp_path):
    env = Environment(fail_at=2)
    path = tmp_path / "step"
    with pytest.raises(RuntimeError, match="interrupted"):
        dispatch(env, [[0, 0], [0, 0]], path)
    assert (path / "000_returned.json").exists()
    assert (path / "001_attempt.json").exists()
    assert not (path / "001_returned.json").exists()
    assert inspect(path) == {"attempts": 2, "returned": 1, "pending": 1, "safe_to_retry": False}
    with pytest.raises(FileExistsError):
        dispatch(env, [[0, 0]], path)
    assert env.calls == 2


def test_tampered_frame_is_rejected(tmp_path):
    path = tmp_path / "step"
    dispatch(Environment(), [[0, 0]], path)
    np.save(path / "000_rgb.npy", np.zeros((224, 224, 3), np.uint8))
    with pytest.raises(ValueError, match="RGB mismatch"):
        inspect(path)


def test_orphan_file_is_rejected(tmp_path):
    path = tmp_path / "step"
    dispatch(Environment(), [[0, 0]], path)
    (path / "orphan").touch()
    with pytest.raises(ValueError, match="unexpected"):
        inspect(path)
