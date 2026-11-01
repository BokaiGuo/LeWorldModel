import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p1_prospective_mechanism_physical_20261101_codex as m


def manifest(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"sealed")
    path = str(source)
    digest = hashlib.sha256(b"sealed").hexdigest()
    role = dict(prefix_raw=np.zeros((5, 2)).tolist(), own_suffix_raw=np.zeros((15, 2)).tolist(),
                cp5_pixel_sha256="a" * 64, cp5_state_identity_sha256="b" * 64,
                cp5_dynamics_sha256="d" * 64)
    return dict(schema=m.NAME + ".input.v1", sources={path: digest}, source_evidence=[path],
        planning_evidence=[path], root=dict(reset_seed=3, requested_reset=[0.] * 7,
        placeholder_goal=[0.] * 7, goal_state=[0.] * 7, root_state=[0.] * 7,
        root_dynamics_sha256="e" * 64,
        warmup_raw=np.zeros((10, 2)).tolist(),
        warmup_rgb=[dict(path=path, pixel_sha256="c" * 64) for _ in range(3)]),
        roles={k: dict(role) for k in m.ROLES})


def test_nine_fixed_branches_and_s_common(tmp_path):
    d = manifest(tmp_path)
    d["roles"]["S"]["own_suffix_raw"] = np.full((15, 2), .25).tolist()
    d["roles"]["G"]["own_suffix_raw"] = np.full((15, 2), -.5).tolist()
    schedule = m.build_schedule(d)
    assert [(x["role"], x["tail"]) for x in schedule] == [(r, t) for r in m.ROLES for t in m.TAILS]
    for entry in schedule:
        suffix = np.asarray(entry["commands"])[5:]
        expected = .25 if entry["tail"] == "common" or (entry["tail"] == "own" and entry["role"] == "S") else (
            -.5 if entry["tail"] == "own" and entry["role"] == "G" else 0.)
        assert np.all(suffix == expected)
    d["roles"]["Z"]["prefix_raw"][0] = [0.1, 0.]
    with pytest.raises(ValueError, match="Z prefix"):
        m.build_schedule(d)


def test_source_hash_and_manifest_tamper_rejected(tmp_path):
    d = manifest(tmp_path)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(d))
    sha = m.file_sha(path)
    m.load_input(path, sha)
    path.write_text(json.dumps(d) + " ")
    with pytest.raises(ValueError, match="manifest hash"):
        m.load_input(path, sha)
    path.write_text(json.dumps(d))
    (tmp_path / "source").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source missing or hash changed"):
        m.load_input(path, sha)


def test_no_missing_evidence_and_invalid_actions(tmp_path):
    d = manifest(tmp_path)
    d["planning_evidence"] = []
    with pytest.raises(ValueError, match="planning evidence"):
        m.build_schedule(d)
    d["planning_evidence"] = d["source_evidence"]
    d["roles"]["G"]["prefix_raw"] = np.full((5, 2), 1.01).tolist()
    with pytest.raises(ValueError, match="outside"):
        m.build_schedule(d)


@pytest.mark.parametrize("case", ["complete", "cp5_mismatch", "early_stop"])
def test_actual_step_seam_binds_cp5_before_suffix(tmp_path, monkeypatch, case):
    state = np.array([100., 100., 100., 100., 0., 0., 0.])
    pixels = np.zeros((224, 224, 3), np.uint8)
    goal = np.zeros(7)

    class Env:
        def __init__(self):
            self.unwrapped = self
            self.latest_action = None
            self.calls = 0
        def reset(self, **kwargs):
            return None, {}
        def step(self, action):
            self.calls += 1
            self.latest_action = action.copy()
            return {"state": state.copy()}, 0., case == "early_stop" and self.calls == 13, False, {"goal_state": goal.copy()}
        def render(self):
            return pixels.copy()
        def _get_obs(self):
            return state.copy()

    monkeypatch.setattr(m.native_source, "attach_goal", lambda *args: None)
    dynamics = {"state": state.tolist(), "fixture": True}
    monkeypatch.setattr(m.native_source, "snapshot", lambda body: dynamics)
    monkeypatch.setattr(m.evaluator, "metrics", lambda *args: {"object_xy": 0.})
    monkeypatch.setattr(m.evaluator.qualification, "native", lambda *args:
                        {"joint_xy_error": 0., "initial_success": False})
    slot = dict(role="S", tail="common", commands=np.zeros((20, 2)).tolist(),
                cp5_pixel_sha256="0" * 64 if case == "cp5_mismatch" else hashlib.sha256(pixels.tobytes()).hexdigest(),
                cp5_state_identity_sha256=m.legacy.panel.saved.spec.digest(state.tolist()),
                cp5_dynamics_sha256=m.legacy.panel.saved.spec.digest(dynamics))
    root = dict(reset_seed=1, requested_reset=state.tolist(), placeholder_goal=goal.tolist(),
                goal_state=goal.tolist(), root_state=state.tolist(),
                root_dynamics_sha256=m.legacy.panel.saved.spec.digest(dynamics),
                warmup_raw=np.zeros((10, 2)).tolist())
    row = m.run_branch(Env(), root, [pixels] * 3, slot, tmp_path)
    assert row["physical_cost"]["warmup_returned"] == 10
    if case == "cp5_mismatch":
        assert row["status"] == "failed" and row["steps"] == 5
        assert "cp5" in row["error"]["message"]
    elif case == "early_stop":
        assert row["status"] == "native_stopped" and row["steps"] == 3
        assert [p["step"] for p in row["checkpoints"]] == [0, 3]
    else:
        assert row["status"] == "completed20" and row["steps"] == 20
        assert [p["step"] for p in row["checkpoints"]] == [0, 5, 10, 15, 20]
