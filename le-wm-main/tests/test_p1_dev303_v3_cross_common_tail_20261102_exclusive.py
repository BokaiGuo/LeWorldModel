import copy
import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability import p1_dev303_v3_cross_common_tail_20261102_exclusive as run


def test_real_sealed_cross_and_source():
    contract = run.build()
    assert len(contract["tail_family"]) == 5
    assert len(contract["branches"]) == 20
    assert [b["prefix_role"] for b in contract["branches"][::5]] == [
        "selected", "zero", "task_greedy_direct", "random"]
    assert [x["tail"] for x in contract["tail_family"]] == [
        "geometry_0", "geometry_1", "geometry_2", "geometry_3", "zero"]
    assert all(x["actions"][:5] == x["actions"][5:] for x in contract["tail_family"])
    assert not np.asarray(contract["tail_family"][-1]["actions"]).any()
    for i in range(5):
        assert all(contract["branches"][j * 5 + i]["tail_sha256"] ==
                   contract["tail_family"][i]["sha256"] for j in range(4))
    assert contract["P1_admission"] is False and contract["independent_confirmation"] is False


def test_tail_source_tamper_fails():
    value = copy.deepcopy(run.base.read(run.TAIL_PREP))
    value["candidates"][1]["actions"][0][0] = 0
    with pytest.raises(ValueError, match="tail source action hash"):
        run.tail_family(value)


class FakeEnv:
    def __init__(self, stop=None):
        self.n = 0
        self.stop = stop
        self.latest_action = np.zeros(2, dtype=np.float32)

    def reset(self, seed):
        return {"state": [0.0] * 5}, {"goal_state": [0.0] * 5, "goal": np.zeros((2, 2, 3), np.uint8)}

    def step(self, action):
        self.n += 1
        self.latest_action = action.copy()
        return ({"state": [0.0] * 5}, 0.0, self.n == self.stop, False,
                {"goal_state": [0.0] * 5, "goal": np.zeros((2, 2, 3), np.uint8)})


def exercise(tmp_path, monkeypatch, stop=None, mismatch=False):
    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    pixsha = hashlib.sha256(pixels.tobytes()).hexdigest()
    state = [0.0] * 5
    warm = [{"step": i, "state": state, "snapshot": {"state": state}, "pixel_sha256": pixsha}
            for i in (0, 5, 10)]
    cps = [{"step": i, "state": state, "snapshot": {"state": state}, "pixel_sha256": pixsha}
           for i in range(1, 6)]
    if mismatch:
        cps[4] = {**cps[4], "pixel_sha256": "0" * 64}
    tail = np.full((10, 2), 0.25, dtype="<f4")
    role = {"arm": "test", "prefix": [[0.0, 0.0]] * 5, "prefix_checkpoints": cps,
            "cp5_pixel_sha256": cps[-1]["pixel_sha256"], "tail": tail.tolist(),
            "tail_sha256": hashlib.sha256(tail.tobytes()).hexdigest()}
    history = {"goal": state, "frames": warm}
    store = run.base.Store(tmp_path / "run")
    row = {"arm": "test", "status": "running", "transitions": [], "observations": [], "error": None}

    def capture(store, env, arm, phase, step, expected=None):
        point = {**store.frame(f"{arm}_{phase}_{step}.png", pixels), "phase": phase,
                 "step": step, "state": state, "snapshot": {"state": state}}
        if expected is not None:
            run.require(all(point[k] == expected[k] for k in ("state", "snapshot", "pixel_sha256")),
                        "checkpoint mismatch")
        store.emit("observation", arm=arm, **point)
        return point

    monkeypatch.setattr(run.replay.base, "capture", capture)
    store.emit("branch_started", arm="test")
    try:
        run.replay.branch(FakeEnv(stop), store, role, history, {"state": state},
                          np.zeros((2, 2, 3), np.uint8), row, capture=capture)
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = {"type": type(exc).__name__, "message": str(exc)}
        store.emit("branch_error", arm="test", error=row["error"])
    store.emit("branch_closed", arm="test", status=row["status"])
    return store, role, row, history


@pytest.mark.parametrize("stop, count, status", [(None, 10, "fixed_horizon_completed"),
                                                (17, 2, "native_early_stop")])
def test_synthetic_replay_and_saved_branch(tmp_path, monkeypatch, stop, count, status):
    store, role, row, history = exercise(tmp_path, monkeypatch, stop=stop)
    result = run.replay.verifier.verify_branch(store.path, role, row, store.events, history)
    assert result["status"] == status and result["tail_returned"] == count


def test_cp5_mismatch_never_dispatches_tail(tmp_path, monkeypatch):
    store, role, row, history = exercise(tmp_path, monkeypatch, mismatch=True)
    result = run.replay.verifier.verify_branch(store.path, role, row, store.events, history)
    assert result["status"] == "failed" and result["tail_returned"] == 0
    assert not any(e["kind"] == "action_dispatch" and e["phase"] == "tail" for e in store.events)


def test_explicit_authority_required():
    with pytest.raises(ValueError, match="authority"):
        run.execute("missing", "missing", "missing", None)
