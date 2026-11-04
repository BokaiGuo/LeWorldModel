import json

import numpy as np
import pytest

from experiments.spectral_reachability import p1_dev305_goal_recapture_live_20261102_exclusive as live


def image(invalid=False):
    out = np.full((224, 224, 3), 255, dtype=np.uint8)
    out[210:224 if invalid else 70, 50:70] = [120, 140, 160] if invalid else 255
    if not invalid:
        out[50:70, 50:70] = [120, 140, 160]
    out[100:110, 100:110] = [50, 100, 230]
    return out


class Env:
    def __init__(self, goal, current, fail=False):
        self.goal, self.current, self.fail = goal, current, fail
        self.closed = False

    def reset(self, seed):
        assert seed == live.SEED
        if self.fail:
            raise RuntimeError("reset failed")
        return {}, {"goal": self.goal}

    def render(self):
        return self.current

    def step(self, *_):
        raise AssertionError("physical step forbidden")

    def close(self):
        self.closed = True


def fake_run(tmp_path, goal, current, *, fail_second=False, wrong_hash=False):
    instances = []

    def factory():
        env = Env(goal, current, fail=fail_second and len(instances) == 1)
        instances.append(env)
        return env

    gh = live.digest(goal.tobytes())
    ch = "bad" if wrong_hash else live.digest(current.tobytes())
    report = live.run(tmp_path / "out", factory, gh, ch)
    assert all(x.closed for x in instances)
    assert json.loads((tmp_path / "out/report.json").read_text()) == report
    return report


def test_guard_blocks_two_real_calls_not_reused_pixels(tmp_path):
    report = fake_run(tmp_path, image(True), image())
    assert report["status"] == "completed"
    assert report["decision"]["reason"] == "goal_visibility_failed"
    assert report["selector_calls"] == 0
    assert report["physical_steps"] == 0
    assert len(report["frames"]) == 3
    assert report["acquisition_metadata"]["capture_ids"][0] != report["acquisition_metadata"]["capture_ids"][1]


def test_valid_goal_still_cannot_dispatch_candidate(tmp_path):
    report = fake_run(tmp_path, image(), image())
    assert report["status"] == "completed"
    assert report["decision"]["status"] == "blocked"
    assert report["decision"]["selector_called"]
    assert report["decision"]["action"] is None


@pytest.mark.parametrize("kwargs", [{"fail_second": True}, {"wrong_hash": True}])
def test_failure_keeps_evidence_and_stops(tmp_path, kwargs):
    report = fake_run(tmp_path, image(), image(), **kwargs)
    assert report["status"] == "engineering_failure"
    assert "decision" not in report


def test_no_overwrite(tmp_path):
    path = tmp_path / "out"
    path.mkdir()
    with pytest.raises(FileExistsError):
        live.run(path, lambda: None, "", "")
