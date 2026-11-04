import numpy as np
import pytest
import json
from pathlib import Path

from experiments.spectral_reachability import run_reacher_pc_p1_fd_common_physical_20261102_codexexclusive as run


def test_branch_order_count_and_common_tail_binding():
    prefixes = np.arange(90, dtype=np.float32).reshape(9, 5, 2) / 1000
    tails = np.arange(100, dtype=np.float32).reshape(5, 10, 2) / 1000
    branches = run.make_branches(prefixes, tails)
    assert len(branches) == 10
    assert [(b["arm"], b["tail_index"]) for b in branches] == [
        (arm, i) for arm in ("selected", "parameter_zero") for i in range(5)]
    for index in range(5):
        a, b = (np.asarray(branches[i]["raw_actions"], dtype=np.float32) for i in (index, index + 5))
        np.testing.assert_array_equal(a[:5], prefixes[4])
        np.testing.assert_array_equal(b[:5], prefixes[0])
        np.testing.assert_array_equal(a[5:], b[5:])
        assert branches[index]["actions_sha256"] == run.digest_actions(a)


def test_bad_actions_fail_closed():
    prefixes = np.zeros((9, 5, 2), dtype=np.float32)
    tails = np.zeros((5, 10, 2), dtype=np.float32)
    tails[0, 0, 0] = 3
    with pytest.raises(ValueError, match="invalid bound"):
        run.make_branches(prefixes, tails)


class FakeEnv:
    def __init__(self, *, stop=None, throws=None):
        self.count = 0
        self.stop = stop
        self.throws = throws

    def step(self, action):
        self.count += 1
        if self.count == self.throws:
            raise RuntimeError("step error")
        return None, 0., self.count == self.stop, False, {}

    def render(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)


@pytest.fixture
def fake_context(monkeypatch):
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    snapshot = {key: "value" for key in ("physics_integration_state", "integration_state_spec",
                "model_binary_sha256", "target_qpos", "target_marker_pos", "rendered_goal_rgb")}
    ctx = {"execution_snapshot": snapshot, "root_frame": image, "goal": image}
    monkeypatch.setattr(run.context_lib, "restore_execution_snapshot", lambda *a, **k: (image, image))
    monkeypatch.setattr(run.context_lib, "capture_execution_snapshot", lambda env: snapshot)
    monkeypatch.setattr(run.common, "record", lambda env: {"qpos": [float(env.count)], "qvel": [0.],
                                                          "target_qpos": [0.], "qpos_distance": float(env.count)})
    return ctx


@pytest.mark.parametrize("stop,expected,steps", [(None, "complete", 15), (3, "stopped", 3), (15, "stopped", 15)])
def test_steps_rgb_and_stop_are_preserved(fake_context, stop, expected, steps):
    result = run.execute_branch(FakeEnv(stop=stop), 1, fake_context, np.zeros((15, 2), np.float32))
    assert result["status"] == expected
    assert result["returned_steps"] == result["attempted_steps"] == steps
    assert len(result["trajectory"]) == steps + 1
    assert run.context_lib.unpack_rgb_observation(result["trajectory"][-1]["rgb"]).shape == (2, 2, 3)
    assert result["trajectory"][1]["action"] == [0., 0.]


def test_error_and_incomplete_snapshot_fail_closed(fake_context):
    result = run.execute_branch(FakeEnv(throws=4), 1, fake_context, np.zeros((15, 2)))
    assert result["status"] == "error" and result["attempted_steps"] == 4 and result["returned_steps"] == 3
    del fake_context["execution_snapshot"]["physics_integration_state"]
    result = run.execute_branch(FakeEnv(), 1, fake_context, np.zeros((15, 2)))
    assert result["status"] == "error" and result["returned_steps"] == 0


def test_stop_survives_render_error(fake_context):
    class BrokenRender(FakeEnv):
        def render(self):
            raise RuntimeError("render error")
    result = run.execute_branch(BrokenRender(stop=1), 1, fake_context, np.zeros((15, 2)))
    assert result["status"] == "error"
    assert result["returned_steps"] == 1
    assert result["stop"] == {"step": 1, "terminated": True, "truncated": False}


def test_actual_sealed_model_action_selection_without_environment():
    root = Path(__file__).resolve().parents[1]
    model = json.loads((root / "results/reacher_pc_fd_suffix_cuda_20261102_v1/report.json").read_text())
    saved = json.loads((root / "results/reacher_pc_fd_suffix_saved_audit_20261102_v1.json").read_text())
    preflight = json.loads((root / "results/reacher_pc_p1_common_tail_preflight_20261102_codexexclusive_02.json").read_text())
    mean, std = run.setup._load_action_stats(Path(preflight["action_stats_path"]), "swm/ReacherDMControl-v0")
    for row, audited in zip(model["rows"], saved["rows"], strict=True):
        prefixes, tails = run.source_actions(row, mean, std, audited)
        assert len(run.make_branches(prefixes, tails)) == 10
        row["compiler_receipt"]["strict_selected_index"] = 4
        with pytest.raises(ValueError, match="selection mismatch"):
            run.source_actions(row, mean, std, audited)
