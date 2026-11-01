import copy

import numpy as np
import pytest

from experiments.spectral_reachability import p1_dev303_cp5_tail_physical_20261102_exclusive as run
from experiments.spectral_reachability import p1_dev303_cp5_tail_physical_saved_20261102_exclusive as saved


def point(step):
    pixels = np.full((224, 224, 3), step, np.uint8)
    state = [float(step)]
    return {"step": step, "state": state, "snapshot": {"state": state},
            "pixel_sha256": run.base.pixel_sha(pixels)}


class FakeEnv:
    def __init__(self, *, bad_action_at=None, stop_at=None):
        self.n = 0
        self.bad_action_at = bad_action_at
        self.stop_at = stop_at
        self.latest_action = None

    def reset(self, seed):
        return {"state": [0.]}, {"goal_state": [9.], "goal": np.zeros((2, 2, 3), np.uint8)}

    def step(self, action):
        self.n += 1
        self.latest_action = action.copy()
        if self.n == self.bad_action_at:
            self.latest_action[0] = 1
        return ({"state": [float(self.n)]}, 0.0, self.n == self.stop_at, False,
                {"goal_state": [9.], "goal": np.zeros((2, 2, 3), np.uint8)})


def capture(store, env, arm, phase, step, expected=None):
    point_data = point(env.n)
    frame = store.frame(f"{arm}_{phase}_{step:02d}.png", np.full((224, 224, 3), env.n, np.uint8))
    value = {**frame, "phase": phase, "step": step, "state": point_data["state"],
             "snapshot": point_data["snapshot"]}
    if expected is not None:
        run.require(all(value[k] == expected[k] for k in ("state", "snapshot", "pixel_sha256")),
                    "checkpoint mismatch")
    store.emit("observation", arm=arm, **value)
    return value


def fixture_data():
    tail = np.full((10, 2), 0.5, dtype="<f4")
    role = {"arm": "rgb_chunk", "prefix": [[0., 0.]] * 5, "tail": tail.tolist(),
            "tail_sha256": run.handoff.digest(tail.tobytes()),
            "prefix_checkpoints": [{**point(i + 10), "step": i} for i in range(1, 6)],
            "cp5_pixel_sha256": point(15)["pixel_sha256"]}
    history = {"goal": [9.], "frames": [point(i) for i in (0, 5, 10)]}
    return role, history


def run_fake(tmp_path, *, mismatch=False, bad_action_at=None, stop_at=None):
    role, history = fixture_data()
    if mismatch:
        role["prefix_checkpoints"][4]["pixel_sha256"] = "wrong"
    store = run.base.Store(tmp_path / "out")
    env = FakeEnv(bad_action_at=bad_action_at, stop_at=stop_at)
    row = {"arm": role["arm"], "status": "running", "transitions": [], "observations": [], "error": None}
    store.emit("branch_started", arm=role["arm"])
    try:
        run.branch(env, store, role, history, {"state": [0.]}, np.zeros((2, 2, 3), np.uint8),
                   row, capture=capture)
    except Exception as exc:
        row.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        store.emit("branch_error", arm=role["arm"], error=row["error"])
    store.emit("branch_closed", arm=role["arm"], status=row["status"])
    return role, history, store, row, env


def test_synthetic_full_and_saved_verification(tmp_path):
    role, history, store, row, env = run_fake(tmp_path)
    assert env.n == 25 and row["status"] == "fixed_horizon_completed"
    assert saved.verify_branch(store.path, role, row, store.events, history)["tail_returned"] == 10
    altered = copy.deepcopy(store.events)
    next(e for e in altered if e["kind"] == "action_dispatch" and e["phase"] == "tail")["action"] = [0., 0.]
    with pytest.raises(ValueError):
        saved.verify_branch(store.path, role, row, altered, history)


@pytest.mark.parametrize("kwargs", [{"mismatch": True}, {"bad_action_at": 12}])
def test_prefix_failure_dispatches_no_tail_and_is_recorded(tmp_path, kwargs):
    role, history, store, row, env = run_fake(tmp_path, **kwargs)
    assert row["status"] == "failed" and env.n <= 15
    assert not any(e["kind"] == "action_dispatch" and e["phase"] == "tail" for e in store.events)
    assert saved.verify_branch(store.path, role, row, store.events, history)["tail_returned"] == 0


def test_early_stop_no_padding(tmp_path):
    role, history, store, row, env = run_fake(tmp_path, stop_at=17)
    assert row["status"] == "native_early_stop" and env.n == 17
    assert saved.verify_branch(store.path, role, row, store.events, history)["tail_returned"] == 2


def test_exclusive_output(tmp_path):
    run.base.Store(tmp_path / "exclusive")
    with pytest.raises(ValueError):
        run.base.Store(tmp_path / "exclusive")


def test_real_sources_contract_is_development_only():
    contract = run.build()
    assert contract["source"]["planner_report_sha256"] == run.handoff.REPORT_SHA
    assert contract["physical_actions_authorized_by_contract"] is False
    assert contract["P1_admission"] is False


def test_saved_report_and_frame_tampering(tmp_path, monkeypatch):
    role, history = fixture_data()
    contract = {"source": {"roles": [role]}}
    prepared = tmp_path / "prepared.json"
    prepared.write_bytes(run.base.canonical(contract) + b"\n")
    prepared_sha = run.base.sha(prepared)
    monkeypatch.setattr(run, "build", lambda: contract)
    monkeypatch.setattr(saved.prior, "source_data", lambda: (None, {"history": history}, None))
    store = run.base.Store(tmp_path / "out")
    store.json("execution_binding.json", {"prepared": str(prepared.resolve()),
                                          "prepare_sha256": prepared_sha})
    store.emit("run_started", prepare_sha256=prepared_sha)
    row = {"arm": role["arm"], "status": "running", "transitions": [], "observations": [], "error": None}
    store.emit("branch_started", arm=role["arm"])
    run.branch(FakeEnv(), store, role, history, {"state": [0.]}, np.zeros((2, 2, 3), np.uint8),
               row, capture=capture)
    store.json(role["arm"] + ".row.json", row)
    store.emit("branch_closed", arm=role["arm"], status=row["status"])
    store.emit("run_finished", errors=[])
    report = {"schema": run.NAME + ".report.v1", "prepare_sha256": prepared_sha,
              "status": "recorded", "rows": [row], "errors": [],
              "journal_sha256": run.base.sha(store.path / "journal.jsonl"),
              "journal_tail": store.events[-1]["sha256"], "P1_admission": False,
              "independent_confirmation": False, "classification": "development_only"}
    store.json("report.json", report)
    report_sha = run.base.sha(store.path / "report.json")
    assert saved.verify(prepared, prepared_sha, store.path, report_sha)["run_status"] == "recorded"
    with pytest.raises(ValueError):
        saved.verify(prepared, prepared_sha, store.path, "0" * 64)
    (store.path / row["observations"][-1]["file"]).write_bytes(b"tampered")
    with pytest.raises(ValueError):
        saved.verify(prepared, prepared_sha, store.path, report_sha)
