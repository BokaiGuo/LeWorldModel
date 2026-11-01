"""CPU fixtures and actual metadata only; never load a neural model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import run_pusht_sampling_comparison_20261027_v2 as v2


def images():
    frames = [np.full((224,224,3), i, dtype=np.uint8) for i in range(3)]
    goal = np.full((224,224,3), 8, dtype=np.uint8)
    # Independently reproduce the real preflight's whole-stack scalar field.
    reference = {"history_rgb_sha256": hashlib.sha256(np.stack(frames).tobytes()).hexdigest(),
                 "goal_rgb_sha256": hashlib.sha256(goal.tobytes()).hexdigest()}
    return frames, goal, reference


def test_actual_upstream_metadata_is_one_sha_not_per_frame():
    source = Path(v2.v1.DEFAULT_COMPILATION)
    if not source.exists():
        pytest.skip("local calibration absent")
    report = json.loads(source.read_text())
    bindings = report["protocol"]["bindings"]
    upstreams = [json.loads(Path(p).read_text()) for p in bindings if Path(p).name == "report.json"]
    rows = next(r["rows"] for r in upstreams if "history_rgb_sha256" in r["rows"][0])
    assert len(rows) == 4
    for row in rows:
        assert type(row["history_rgb_sha256"]) is str
        assert len(row["history_rgb_sha256"]) == 64
        assert type(row["goal_rgb_sha256"]) is str


def test_whole_stack_accepts_real_structure_not_v1_list():
    frames, goal, ref = images()
    assert v2.verify_rgb_context(frames, goal, ref) == ref
    assert [v2.sha_rgb(f) for f in frames] != ref["history_rgb_sha256"]
    with pytest.raises(ValueError, match="single whole-stack"):
        v2.verify_rgb_context(frames, goal, dict(ref, history_rgb_sha256=[v2.sha_rgb(f) for f in frames]))


@pytest.mark.parametrize("tamper", ["pixel", "order", "goal", "reference", "dtype"])
def test_stack_identity_does_not_silently_pass(tamper):
    frames, goal, ref = images()
    if tamper == "pixel":
        frames[1][0,0,0] = 100
    elif tamper == "order":
        frames.reverse()
    elif tamper == "goal":
        goal[0,0,0] = 100
    elif tamper == "reference":
        ref["history_rgb_sha256"] = "0"*64
    else:
        frames = [f.astype(float) for f in frames]
    with pytest.raises(ValueError):
        v2.verify_rgb_context(frames, goal, ref)


class Env:
    """Actual shared RGB guard/executor runs on this non-neural fixture."""
    def __init__(self, disappear=100, terminate=100, fail=False, lie=False):
        self.count, self.disappear, self.terminate = 0, disappear, terminate
        self.fail, self.lie = fail, lie
        self.commands = []
        self.unwrapped = self
    def render(self):
        rgb = np.full((224,224,3), 255, np.uint8)
        if self.count < self.disappear:
            rgb[20,20] = [65,105,225]
        return rgb
    def state(self):
        return np.asarray([float(self.count),0.,0.,0.,0.,0.,0.])
    def step(self, command):
        self.commands.append(command.copy())
        if self.fail:
            raise RuntimeError("fixture step failure")
        self.count += 1
        return {}, float(self.count), self.count >= self.terminate, False, {}
    def eval_state(self, goal, state):
        difference = abs(goal[4]-state[4])
        success = np.linalg.norm(goal[:4]-state[:4]) < 20 and min(difference,2*np.pi-difference) < np.pi/9
        return (not success if self.lie else success), np.linalg.norm(goal-state)


def run(env):
    return v2.guarded_branch(env, np.tile([.1,0.], (10,1)), root_id="fixture-root",
        goal=np.zeros(7), state_of=lambda e: e.state(),
        pose_label=lambda s,g: {"object_distance": float(np.linalg.norm(s[2:4]-g[2:4]))})


@pytest.mark.parametrize("disappear,steps,status", [(0,0,"observation_unavailable"),
    (2,2,"observation_unavailable"), (100,10,"completed")])
def test_guarded_branch_keeps_aborts_labels_pose_and_exact_prefix(disappear, steps, status):
    env = Env(disappear=disappear)
    result = run(env)
    assert result["status"] == status
    assert len(env.commands) == result["completed_commands"] == len(result["trace"]) == steps
    assert result["remaining_commands_not_executed"] == 10-steps
    assert result["full_horizon_completed"] is (steps == 10)
    assert result["trace_matches_sealed_prefix"] is True
    assert result["guard"]["root_id"] == "fixture-root"
    assert "official_success" in result["official"] and "object_distance" in result["pose"]
    assert "official" in result["initial"] and "pose" in result["initial"]
    assert all("official" in t and "pose" in t and "rgb_sha256" in t for t in result["trace"])


def test_termination_is_retained_not_padded_as_full_horizon():
    result = run(Env(terminate=3))
    assert result["status"] == "environment_terminated"
    assert result["completed_commands"] == 3
    assert not result["full_horizon_completed"]


def test_executor_failure_is_retained_without_retry():
    env = Env(fail=True)
    result = run(env)
    assert result["status"] == "execution_failed"
    assert result["attempted_commands"] == len(env.commands) == 1
    assert result["completed_commands"] == 0
    assert not result["full_horizon_completed"]
    assert "official" in result and "pose" in result


def test_false_official_label_cannot_pass():
    with pytest.raises(ValueError, match="installed evaluator"):
        run(Env(lie=True))


def test_forged_completed_receipt_without_observations_rejected(monkeypatch):
    def fake(env, commands, **kwargs):
        return {"status": "completed", "completed_commands": 10, "attempted_commands": 10,
            "planned_commands": 10, "remaining_commands_not_executed": 0,
            "guard": {"root_id": "fixture-root", "primitive_commands_authorized": 10},
            "terminal_rgb": env.render()}
    monkeypatch.setattr(v2.visible, "execute", fake)
    with pytest.raises(ValueError, match="receipt/observed trace"):
        run(Env())


def test_v1_plans_report_never_accepted_as_v2(tmp_path):
    with pytest.raises(ValueError, match="V2 protocol-bound"):
        v2.validate_plans(tmp_path, {}, {}, {"protocol_sha256":"x"},
            {"status":"plans_sealed", "gates":{"ok":True}, "rows":[]}, None)
