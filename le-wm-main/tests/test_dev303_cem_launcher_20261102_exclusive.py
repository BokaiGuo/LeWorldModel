import copy
import json

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import dev303_cem_launcher_20261102_exclusive as run
from experiments.spectral_reachability import dev303_native_host_bridge_20261102_exclusive as bridge


def test_contract_scopes_and_drift(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"original")
    contract = run.contract_from_bindings({str(source): run.sha(source)}, "b" * 64)
    assert contract["classification"] == "dev303_exploratory"
    assert not contract["equal_budget_claim"]
    assert not contract["physical_execution_authorized_by_contract"]
    run.verify_bindings(contract)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="binding"):
        run.verify_bindings(contract)


def test_exclusive_contract_and_external_sha(tmp_path):
    path = tmp_path / "contract.json"
    digest = run.write_new(path, {"x": 1})
    assert run.sha(path) == digest
    with pytest.raises(FileExistsError):
        run.write_new(path, {"x": 2})


def test_saved_journal_and_partial_fail_closed(tmp_path):
    store = run.base.Store(tmp_path / "run")
    store.emit("run_started", prepare_sha256="a" * 64)
    store.emit("run_finished", status="partial_or_failed")
    rows = run.load_journal(store.path / "journal.jsonl",
                            run.sha(store.path / "journal.jsonl"),
                            store.events[-1]["sha256"])
    assert len(rows) == 2
    with pytest.raises(ValueError, match="complete"):
        run.verify_sequence(rows, seed=42)
    path = store.path / "journal.jsonl"
    path.write_bytes(path.read_bytes().replace(b"run_started", b"run_starteD"))
    with pytest.raises(ValueError, match="journal"):
        run.load_journal(path, "0" * 64, store.events[-1]["sha256"])


def test_replay_candidate_bank_and_selection_detect_tamper():
    events = []
    run.cem.optimize(3, lambda x: float(np.square(x).sum()), seed=42, emit=events.append)
    run.verify_optimizer(events, 3, 42)
    corrupted = copy.deepcopy(events)
    next(e for e in corrupted if e["type"] == "selection")["index"] = 1
    with pytest.raises(ValueError, match="optimizer"):
        run.verify_optimizer(corrupted, 3, 42)


def test_call_accounting_uses_attempts_not_just_returns():
    rows = [{"type": "cost_attempt", "kind": "predict"},
            {"type": "cost_failure", "kind": "predict", "elapsed_seconds": 0.0}]
    result = run.costs_from_events(rows)
    assert result["predict"] == {"attempted": 1, "returned": 0, "failed": 1}


@pytest.mark.parametrize("stop_at", [None, 3, 8])
def test_complete_or_early_stop_synthetic_journal(stop_at):
    class Model:
        def action_encoder(self, x):
            return x
        def predict(self, states, actions):
            return states
    events = []
    def step(a):
        step.count += 1
        return a, step.count == stop_at
    step.count = 0
    host = bridge.Host(Model(), lambda _: torch.zeros(2),
                       lambda _: np.zeros((4, 4, 3), np.uint8), step,
                       [torch.zeros(2) for _ in range(3)], torch.zeros(2),
                       np.zeros((10, 2), np.float32), torch.zeros(2), torch.ones(2),
                       events.append, "a" * 64)
    host.run(seed=42)
    rows = [{"kind": "run_started"}] + [{"kind": "bridge", "event": e} for e in events]
    rows.append({"kind": "run_finished",
                 "status": "horizon_completed" if stop_at is None else "native_early_stop"})
    result = run.verify_sequence(rows, 42)
    assert result["control_returned"] == (stop_at or 15)
    damaged = copy.deepcopy(rows)
    event = next(r["event"] for r in damaged if r["kind"] == "bridge"
                 and r["event"]["type"] == "control_return")
    event["action"] = [1.0, 1.0]
    with pytest.raises(ValueError):
        run.verify_sequence(damaged, 42)


def test_reset_warmup_replays_zero_actions_and_exact_checkpoints(tmp_path, monkeypatch):
    pixels = [np.full((224, 224, 3), i, np.uint8) for i in range(3)]
    goal_rgb = np.zeros((224, 224, 3), np.uint8)
    state = [0.0] * 5
    goal = [1.0] * 5
    class Env:
        def reset(self, seed):
            assert seed == run.source.SEED
            self.n = 0
            return {"state": state}, {"goal_state": goal, "goal": goal_rgb}
        def step(self, a):
            self.n += 1
            self.latest_action = a
            return ({"state": state}, 0.0, False, False,
                    {"goal_state": goal, "goal": goal_rgb})
    frames = [{"step": i} for i in (0, 5, 10)]
    receipt = {"history": {"frames": frames, "goal": goal, "final_state": state}}
    arrays = {"rgb_history": np.stack(pixels), "goal_rgb": goal_rgb}
    store = run.base.Store(tmp_path / "warmup")
    def capture(s, env, arm, phase, step, expected):
        assert env.n == step and expected["step"] == step
        return s.frame(f"warmup_{step}.png", pixels[step // 5])
    monkeypatch.setattr(run.base, "capture", capture)
    result, _ = run._warmup(Env(), store, arrays, receipt, {"state": state})
    assert all(np.array_equal(a, b) for a, b in zip(result, pixels))
    assert sum(e["kind"] == "warmup_return" for e in store.events) == 10


def test_warmup_rejects_early_stop(tmp_path, monkeypatch):
    goal_rgb = np.zeros((224, 224, 3), np.uint8)
    class Env:
        def reset(self, seed):
            return {"state": [0.]}, {"goal_state": [1.], "goal": goal_rgb}
        def step(self, a):
            self.latest_action = a
            return {"state": [0.]}, 0., True, False, {"goal_state": [1.], "goal": goal_rgb}
    store = run.base.Store(tmp_path / "stop")
    monkeypatch.setattr(run.base, "capture", lambda s, *a: s.frame("frame.png", goal_rgb))
    receipt = {"history": {"frames": [{"step": i} for i in (0, 5, 10)],
                           "goal": [1.], "final_state": [0.]}}
    with pytest.raises(ValueError, match="warmup early stop"):
        run._warmup(Env(), store, {"goal_rgb": goal_rgb,
                                   "rgb_history": np.stack([goal_rgb] * 3)},
                    receipt, {"state": [0.]})


def test_full_saved_verifier_and_frame_tamper(tmp_path, monkeypatch):
    prepared = tmp_path / "prepare.json"
    prepared.write_text("{}")
    prepare_sha = run.sha(prepared)
    pixels = np.zeros((224, 224, 3), np.uint8)
    pixel_sha = run.base.pixel_sha(pixels)
    state = [0.0] * 5
    snapshot = {"state": state}
    refs = [{"step": i, "state": state, "snapshot": snapshot, "pixel_sha256": pixel_sha}
            for i in (0, 5, 10)]
    contract = {"cem_seed": 42, "expected_model_hash": "m" * 64,
                "source_pixels": [pixel_sha] * 4, "source_closure_scope": "fixture"}
    monkeypatch.setattr(run, "check_prepare", lambda *_: contract)
    monkeypatch.setattr(run.source, "source_data",
                        lambda: ({}, {"history": {"frames": refs}}, {}))
    store = run.base.Store(tmp_path / "run")
    store.json("execution_binding.json", {"prepared": str(prepared.resolve()),
                                          "prepare_sha256": prepare_sha})
    store.emit("run_started", prepare_sha256=prepare_sha)
    for resource, count in (("reset", 1), ("warmup_step", 10), ("warmup_render", 3),
                            ("initial_rgb_encode", 4)):
        for _ in range(count):
            store.emit("setup_attempt", resource=resource)
            store.emit("setup_return", resource=resource, wall_seconds=0.)
    def warm_frame(i):
        frame = store.frame(f"warm{i}.png", pixels)
        store.emit("observation", arm="cem", **frame, phase="warmup", step=i,
                   state=state, snapshot=snapshot)
    warm_frame(0)
    for i in range(1, 11):
        store.emit("warmup_dispatch", step=i, action=[0., 0.])
        store.emit("warmup_return", step=i, action=[0., 0.], terminated=False, truncated=False)
        if i in (5, 10):
            warm_frame(i)
    goal = store.frame("cem_goal.png", pixels)
    store.emit("goal_frame", **goal)
    for role in ("observed0", "observed5", "observed10", "goal"):
        name = "latent_" + role + ".json"
        digest = store.json(name, {"role": role, "pixel_sha256": pixel_sha, "values": [0.] * 192})
        store.emit("initial_latent_saved", role=role, file=name, sha256_file=digest,
                   pixel_sha256=pixel_sha)
    def emit(event):
        store.emit("bridge", event=event)
    def capture(step):
        frame = store.frame("cp5.png", pixels)
        store.emit("observation", arm="cem", **frame, phase="control", step=step,
                   state=state, snapshot=snapshot)
        return pixels
    def encode(_):
        digest = store.json("latent_cp5.json", {"role": "cp5", "pixel_sha256": pixel_sha,
                                                "values": [0.] * 192})
        store.emit("cp5_latent_saved", file="latent_cp5.json", sha256_file=digest,
                   pixel_sha256=pixel_sha)
        return torch.zeros(2)
    class Model:
        def action_encoder(self, x):
            return x
        def predict(self, states, actions):
            return states
    host = bridge.Host(Model(), encode, capture, lambda a: (a, False),
                       [torch.zeros(2) for _ in range(3)], torch.zeros(2),
                       np.zeros((10, 2), np.float32), torch.zeros(2), torch.ones(2),
                       emit, pixel_sha)
    host.run(seed=42)
    store.emit("run_finished", status="horizon_completed", errors=[])
    report = {"schema": run.NAME + ".report.v1", "status": "horizon_completed",
              "errors": [], "prepare_sha256": prepare_sha, "control_returned": 15,
              "model_hash_before": "m" * 64, "model_hash_after": "m" * 64,
              "journal_sha256": run.sha(store.path / "journal.jsonl"),
              "journal_tail": store.events[-1]["sha256"], "equal_budget_claim": False,
              "independent_confirmation": False, "P1_admission": False}
    report_sha = store.json("report.json", report)
    assert run.verify(prepared, prepare_sha, store.path, report_sha)["nodes"] == 2
    (store.path / "latent_cp5.json").write_text("{}")
    with pytest.raises(ValueError, match="CP5 latent binding"):
        run.verify(prepared, prepare_sha, store.path, report_sha)
