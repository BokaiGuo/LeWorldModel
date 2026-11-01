import numpy as np
import pytest

from experiments.spectral_reachability import p2_shared_capture_launch_20261102_exclusive as launch
from test_p2_shared_rgb_pair_wiring_20261102_codexexclusive import Adapter, inputs, scope


def test_contract_detects_file_and_source_drift(tmp_path, monkeypatch):
    contract = tmp_path / "contract.json"
    monkeypatch.setattr(launch, "build", lambda: {"fixed": 1})
    digest = launch.write_new(contract, {"fixed": 1})
    assert launch.verify_contract(contract, digest) == {"fixed": 1}
    with pytest.raises(ValueError, match="exclusive"):
        launch.write_new(contract, {"fixed": 1})
    monkeypatch.setattr(launch, "build", lambda: {"fixed": 2})
    with pytest.raises(ValueError, match="drift"):
        launch.verify_contract(contract, digest)
    contract.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        launch.verify_contract(contract, digest)


def test_saved_pair_event_wiring_and_tamper(tmp_path, monkeypatch):
    frames, goal, actions = inputs()
    bundle = {"base": 0, "frames": [{"pixel_sha256": launch.wire.pixel_sha(x)} for x in frames],
              "goal": {"pixel_sha256": launch.wire.pixel_sha(goal)}}
    monkeypatch.setattr(launch.builder, "verify", lambda *_: {"status": "fixture"})
    monkeypatch.setattr(launch.wire, "load_capture", lambda *_: (bundle, frames, goal, actions))
    monkeypatch.setattr(launch.wire.checkpoint, "native_metadata",
                        lambda: {"mean": [0, 0], "std": [1, 1]})
    store = launch.wire.store_api.Store(tmp_path / "run")
    store.emit("run_started", bundle_sha256=launch.CAPTURE_SHA)
    result = launch.wire.run_pair(
        frames, goal, actions, np.zeros(2), np.ones(2),
        lambda x: np.full(192, x[0, 0, 0], np.float32),
        Adapter, scope, lambda event: store.emit("pair_event", event=event), base=0,
    )
    pair_sha = store.json("pair.json", result)
    store.emit("run_finished", status="completed", pair_sha256=pair_sha)
    report = {"schema": launch.wire.NAME + ".report.v1", "status": "completed",
              "bundle_sha256": launch.CAPTURE_SHA, "pair_sha256": pair_sha,
              "journal_sha256": launch.wire.checkpoint.sha(store.path / "journal.jsonl"),
              "journal_tail": store.events[-1]["sha256"],
              "model_tensor_sha256": launch.saved.MODEL_TENSOR_SHA,
              "source_authentication": "capture_bundle_producer_not_authenticated",
              "physical_execution": False, "equal_total_compute": False,
              "independent_confirmation": False, "P1_admission": False, "P2_admission": False}
    report_sha = store.json("report.json", report)
    assert launch.audit(store.path, report_sha)["status"] == "verified_saved_pair_known_root_not_admitted"
    events = [r["event"] for r in store.events if r["kind"] == "pair_event"]
    next(e for e in events if e["kind"] == "branch")["record"]["base"] = 99
    with pytest.raises((ValueError, KeyError)):
        launch.verify_branch_events(result, events, 0)
    with pytest.raises(ValueError, match="report SHA"):
        launch.audit(store.path, "0" * 64)
