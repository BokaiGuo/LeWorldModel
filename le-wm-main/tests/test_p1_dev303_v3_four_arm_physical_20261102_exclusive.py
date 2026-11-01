import copy
import hashlib
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p1_dev303_v3_four_arm_physical_20261102_exclusive as run
from experiments.spectral_reachability import p1_dev303_v3_four_arm_physical_saved_20261102_exclusive as saved


class FakeEnv:
    """Synthetic only: no native environment, model, GPU, or physics."""
    def __init__(self, stop=None, corrupt=None):
        self.n = 0
        self.stop = stop
        self.corrupt = corrupt
        self.latest_action = np.zeros(2, np.float32)

    def reset(self, seed):
        assert seed == run.prior.SEED
        return {"state": [0.0]}, {"goal_state": [0.0], "goal": np.zeros((2, 2, 3), np.uint8)}

    def step(self, action):
        self.n += 1
        self.latest_action = action.copy()
        if self.n == self.corrupt:
            self.latest_action[0] += 0.25
        return ({"state": [0.0]}, 0.0, self.n == self.stop, False,
                {"goal_state": [0.0], "goal": np.zeros((2, 2, 3), np.uint8)})


def synthetic_branch(tmp_path, monkeypatch, stop=None, corrupt=None):
    pixels = np.zeros((224, 224, 3), np.uint8)
    pixel_sha = hashlib.sha256(pixels.tobytes()).hexdigest()
    points = [{"step": i, "state": [0.0], "snapshot": {"state": [0.0]},
               "pixel_sha256": pixel_sha} for i in (0, 5, 10)]
    history = {"goal": [0.0], "final_state": [0.0], "frames": points}
    spec = {"role": "selected", "actions": [[0.25, -0.5]] * 5}
    row = {"arm": "selected", "status": "running", "transitions": [], "observations": [], "error": None}
    store = run.base.Store(tmp_path / "run")

    def capture(store, env, arm, phase, step, expected=None):
        frame = store.frame(f"{arm}_{phase}_{step:02d}.png", pixels)
        point = {**frame, "phase": phase, "step": step, "state": [0.0], "snapshot": {"state": [0.0]}}
        if expected is not None:
            assert all(point[k] == expected[k] for k in ("state", "snapshot", "pixel_sha256"))
        store.emit("observation", arm=arm, **point)
        return point

    monkeypatch.setattr(run.base, "capture", capture)
    store.emit("branch_started", arm="selected")
    try:
        run.branch(FakeEnv(stop, corrupt), store, spec, history, {"state": [0.0]},
                   np.zeros((2, 2, 3), np.uint8), row)
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = {"type": type(exc).__name__, "message": str(exc)}
        store.emit("branch_error", arm="selected", error=row["error"])
    store.emit("branch_closed", arm="selected", status=row["status"])
    return store.path, spec, row, store.events, history


@pytest.mark.parametrize("stop,expected,count", [(None, "fixed_horizon_completed", 5),
                                                (12, "native_early_stop", 2)])
def test_synthetic_saved_branch(tmp_path, monkeypatch, stop, expected, count):
    directory, spec, row, events, history = synthetic_branch(tmp_path, monkeypatch, stop=stop)
    result = saved.verify_branch(directory, spec, row, events, history)
    assert (result["status"], result["prefix_returned"]) == (expected, count)


def test_failed_action_is_not_filled(tmp_path, monkeypatch):
    directory, spec, row, events, history = synthetic_branch(tmp_path, monkeypatch, corrupt=12)
    result = saved.verify_branch(directory, spec, row, events, history)
    assert result["status"] == "failed"
    assert result["prefix_returned"] == 1
    assert row["error"]["message"] == "actual action mismatch"


def test_tampered_action_rejected(tmp_path, monkeypatch):
    directory, spec, row, events, history = synthetic_branch(tmp_path, monkeypatch)
    events = copy.deepcopy(events)
    next(e for e in events if e["kind"] == "action_dispatch" and e["phase"] == "prefix")["action"][0] = 0
    with pytest.raises(ValueError, match="frozen dispatch"):
        saved.verify_branch(directory, spec, row, events, history)


def test_actual_contract_and_explicit_authority():
    contract = run.build()
    assert [(s["role"], s["index"]) for s in contract["arms"]] == list(run.audit.ROLES)
    assert contract["ledger_verified"] is False
    assert contract["equal_budget_claim"] is False
    assert contract["P1_admission"] is False
    with pytest.raises(ValueError, match="authority"):
        run.execute("missing", "missing", "missing", authority=None)


def test_source_drift_blocks(monkeypatch):
    def bad():
        raise ValueError("report SHA mismatch")
    monkeypatch.setattr(run.audit, "audit_local", bad)
    with pytest.raises(ValueError, match="SHA mismatch"):
        run.build()
