import io

import numpy as np
import pytest

from experiments.spectral_reachability import p1_dev303_feedback_live_20261102_codex_exclusive as run
from experiments.spectral_reachability import p1_dev303_feedback_live_saved_20261102_codex_exclusive as saved


def pixels():
    image = np.zeros((224, 224, 3), np.uint8)
    image[98:103, 18:23] = run.feedback.rgb.BLUE
    image[98:103, 80:85] = run.feedback.rgb.BLOCK
    return image


def point(step):
    state = [float(step)]
    return {"step": step, "state": state, "snapshot": {"state": state},
            "pixel_sha256": run.base.pixel_sha(pixels())}


class FakeEnv:
    def __init__(self):
        self.n = 0
        self.latest_action = None

    def reset(self, seed):
        return {"state": [0.]}, {"goal_state": [9.], "goal": np.zeros((2, 2, 3), np.uint8)}

    def step(self, action):
        self.n += 1
        self.latest_action = action.copy()
        return ({"state": [float(self.n)]}, 0., False, False,
                {"goal_state": [9.], "goal": np.zeros((2, 2, 3), np.uint8)})


def capture(store, env, arm, phase, step, expected=None):
    info = point(env.n)
    frame = store.frame(f"{arm}_{phase}_{step:02d}.png", pixels())
    result = {**frame, "phase": phase, "step": step, "state": info["state"], "snapshot": info["snapshot"]}
    if expected is not None:
        run.require(all(result[k] == expected[k] for k in ("state", "snapshot", "pixel_sha256")),
                    "checkpoint mismatch")
    store.emit("observation", arm=arm, **result)
    return result


class FakeScorer(run.feedback.CUDARealScorer):
    def __init__(self, model, task, expected_hash, store, arm, replan):
        self.store, self.arm, self.replan, self.iteration = store, arm, replan, 0
        self.counts = {"rgb_encode_calls": 0, "action_encoder_calls": 0,
                       "predict_calls": 0, "predict_samples": 0}
        self.hooks = {"action_encoder": 0, "predict": 0}
        self.rgb_encoder_hook = 0

    def begin(self, frames, actions):
        self.counts["rgb_encode_calls"] = 3
        self.rgb_encoder_hook = 3

    def score(self, variants):
        self.counts["action_encoder_calls"] += 2
        self.counts["predict_calls"] += 2
        self.counts["predict_samples"] += 640
        self.hooks["action_encoder"] += 2
        self.hooks["predict"] += 2
        scores = np.ones((64, 5))
        scores[0] = 0
        buffer = io.BytesIO()
        np.savez_compressed(buffer, variants=variants, scores=scores)
        raw = buffer.getvalue()
        name = f"{self.arm}_plan_{self.replan:02d}_iteration_{self.iteration:02d}.npz"
        with (self.store.path / name).open("xb") as stream:
            stream.write(raw)
        self.store.emit("score_batch", arm=self.arm, replan=self.replan, iteration=self.iteration,
                        file=name, file_sha256=run.feedback.sha(raw),
                        counts=dict(self.counts), hooks=dict(self.hooks),
                        rgb_encoder_hook=self.rgb_encoder_hook)
        self.iteration += 1
        return scores

    def finish(self):
        return dict(self.counts)

    def close(self):
        pass


def fixture_data():
    tail = np.zeros((10, 2), dtype="<f4")
    role = {"arm": "rgb_chunk", "prefix": [[0., 0.]] * 5, "tail": tail.tolist(),
            "tail_sha256": run.feedback.sha(tail.tobytes()),
            "prefix_checkpoints": [{**point(10+i), "step": i} for i in range(1, 6)],
            "cp5_pixel_sha256": point(15)["pixel_sha256"]}
    source = {"history": {"goal": [9.], "frames": [point(i) for i in (0, 5, 10)]}}
    return role, source


def synthetic_branch(tmp_path, monkeypatch, mismatch=False):
    role, source = fixture_data()
    if mismatch:
        role["prefix_checkpoints"][4]["pixel_sha256"] = "bad"
    monkeypatch.setattr(run.base, "capture", capture)
    monkeypatch.setattr(run, "RecordedScorer", FakeScorer)
    store = run.base.Store(tmp_path / "out")
    row = {"arm": role["arm"], "status": "running", "transitions": [],
           "observations": [], "decisions": [], "error": None}
    store.emit("branch_started", arm=role["arm"])
    env = FakeEnv()
    try:
        run.branch(env, store, role, source, {"state": [0.]}, np.zeros((2, 2, 3), np.uint8),
                   None, None, None, row)
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = {"type": type(exc).__name__, "message": str(exc)}
        store.emit("branch_error", arm=role["arm"], error=row["error"])
    store.emit("branch_closed", arm=role["arm"], status=row["status"])
    return role, source, store, row, env


def test_synthetic_full_feedback_and_saved_reconstruction(tmp_path, monkeypatch):
    role, source, store, row, env = synthetic_branch(tmp_path, monkeypatch)
    assert row["status"] == "feedback_limit_completed" and env.n == 20
    assert len([e for e in store.events if e["kind"] == "score_batch"]) == 25
    assert saved.verify_branch(store.path, role, row, store.events, source["history"])["decisions"] == 5
    score_event = next(e for e in store.events if e["kind"] == "score_batch")
    (store.path / score_event["file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        saved.verify_branch(store.path, role, row, store.events, source["history"])


def test_mismatch_stops_before_model_and_feedback(tmp_path, monkeypatch):
    role, source, store, row, env = synthetic_branch(tmp_path, monkeypatch, mismatch=True)
    assert row["status"] == "failed" and env.n == 15
    assert not any(e["kind"] in ("planning_started", "score_batch") for e in store.events)
    assert saved.verify_branch(store.path, role, row, store.events, source["history"])["status"] == "failed"


def test_model_failure_keeps_partial_cost_and_stops(tmp_path, monkeypatch):
    class FailingScorer(FakeScorer):
        def score(self, variants):
            if self.iteration == 2:
                raise RuntimeError("synthetic model failure")
            return super().score(variants)

    role, source = fixture_data()
    monkeypatch.setattr(run.base, "capture", capture)
    monkeypatch.setattr(run, "RecordedScorer", FailingScorer)
    store = run.base.Store(tmp_path / "out")
    row = {"arm": role["arm"], "status": "running", "transitions": [],
           "observations": [], "decisions": [], "error": None}
    store.emit("branch_started", arm=role["arm"])
    env = FakeEnv()
    try:
        run.branch(env, store, role, source, {"state": [0.]}, np.zeros((2, 2, 3), np.uint8),
                   None, None, None, row)
    except RuntimeError as exc:
        row.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        store.emit("branch_error", arm=role["arm"], error=row["error"])
    store.emit("branch_closed", arm=role["arm"], status=row["status"])
    partial = [e for e in store.events if e["kind"] == "planning_partial"]
    assert env.n == 15 and len(partial) == 1 and partial[0]["counts"]["predict_calls"] == 4
    assert saved.verify_branch(store.path, role, row, store.events, source["history"])["status"] == "failed"


def test_contract_no_execution_authorized():
    contract = run.build()
    assert not contract["P1_admission"] and not contract["physical_execution_authorized_by_contract"]
    assert not contract["cuda_execution_authorized_by_contract"]


def test_saved_top_level_verification(tmp_path, monkeypatch):
    role, source = fixture_data()
    contract = {"replay": {"source": {"roles": [role]}}}
    prepared = tmp_path / "contract.json"
    prepared.write_bytes(run.base.canonical(contract) + b"\n")
    prep_sha = run.base.sha(prepared)
    feedback_prep = tmp_path / "feedback.json"
    feedback_prep.write_bytes(run.base.canonical({"model_hash": {"test": "fake"}}) + b"\n")
    monkeypatch.setattr(run, "build", lambda: contract)
    monkeypatch.setattr(run, "FEEDBACK_PREP", feedback_prep)
    monkeypatch.setattr(saved.prior, "source_data", lambda: (None, source, None))
    monkeypatch.setattr(run.base, "capture", capture)
    monkeypatch.setattr(run, "RecordedScorer", FakeScorer)
    store = run.base.Store(tmp_path / "out")
    store.json("execution_binding.json", {"prepared": str(prepared.resolve()), "prepare_sha256": prep_sha})
    store.emit("run_started", prepare_sha256=prep_sha)
    store.emit("model_loaded", host={"device": "cuda:0"}, model_hash={"test": "fake"})
    row = {"arm": role["arm"], "status": "running", "transitions": [], "observations": [],
           "decisions": [], "error": None}
    store.emit("branch_started", arm=role["arm"])
    run.branch(FakeEnv(), store, role, source, {"state": [0.]}, np.zeros((2, 2, 3), np.uint8),
               None, None, None, row)
    store.json(role["arm"] + ".row.json", row)
    store.emit("branch_closed", arm=role["arm"], status=row["status"])
    store.emit("model_closed", model_hash={"test": "fake"})
    store.emit("run_finished", errors=[])
    report = {"schema": run.NAME + ".report.v1", "prepare_sha256": prep_sha, "rows": [row],
              "errors": [], "status": "recorded", "journal_sha256": run.base.sha(store.path / "journal.jsonl"),
              "journal_tail": store.events[-1]["sha256"], "P1_admission": False,
              "independent_confirmation": False, "classification": "development_only",
              "full_cost_certified": False}
    store.json("report.json", report)
    report_sha = run.base.sha(store.path / "report.json")
    verified = saved.verify(prepared, prep_sha, store.path, report_sha)
    assert verified["status"] == "saved_ledger_verified" and not verified["full_cost_certified"]
    with pytest.raises(ValueError):
        saved.verify(prepared, prep_sha, store.path, "0" * 64)
