import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import codex_sra_online_run_bank_20261030 as s
from experiments.spectral_reachability import codex_online_model_action_20261030 as online

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_test", ROOT / "tests/test_codex_native_runtime_meter_20261030.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
t = helpers.t
A, B = t.A, t.B


def setup(tmp_path):
    files = {name: {"path": str(s.DIR / name), "sha256": s.file_sha(s.DIR / name)}
             for name in s.REQUIRED}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"schema": "codex_sra_online_sources_v1",
                                         "files": files, "action_units": "fixture raw"}))
    manifest_sha = s.file_sha(manifest_path)
    target_path = tmp_path / "target.json"
    target_path.write_text(json.dumps({"goal_rgb_sha256": "c" * 64, "goal_vectors": [[0.] * 192],
                                       "scoring": {"metric": {"projection": np.eye(192).tolist()}}}))
    sources = {}
    for name in t.base.IDENTITY:
        path = target_path if name == "goal_sha256" else tmp_path / name
        if path != target_path:
            path.write_bytes(name.encode())
        sources[name] = {"path": str(path), "sha256": s.file_sha(path)}
    identity = {k: v["sha256"] for k, v in sources.items()}
    config = {"identities": {a: copy.deepcopy(identity) for a in t.base.ARMS},
              "initial_frames": {a: [online.sha(f) for f in t.FRAMES] for a in t.base.ARMS},
              "initial_actions": {a: [copy.deepcopy(t.PACKET), copy.deepcopy(t.PACKET)] for a in t.base.ARMS},
              "horizon": 5, "packet_steps": 5,
              "caps": {a: {k: 5000 for k in t.base.AXES} for a in t.base.ARMS}}
    j = t.jmod.Journal(tmp_path / "journal", config)
    return j, sources, {"path": str(manifest_path), "sha256": manifest_sha}, target_path


def bank():
    labels = (["zero", "dls_scale_.25", "dls_scale_.5", "dls_scale_1"]
              + [f"mode_{i}_radius_{r}_sign_{sign}" for i in range(9)
                 for r in range(2) for sign in (-1, 1)])
    return {"commands": np.zeros((40, 2, 5, 2), dtype=np.float32).tolist(),
            "logits": np.zeros((40, 20)).tolist(), "labels": labels}


def compile_fixture(model, device, ready, compiled, replay, emit):
    for _ in range(4):
        model.predict(np.zeros((1, 3, 192)), model.action_encoder(np.zeros((1, 3, 10))))
    emit({"kind": "jacobian", "completed": True})
    return bank(), [{"stage": "fixture"}]


def replay_fixture(model):
    def replay(context, actions):
        model.predict(np.zeros((1, 3, 192)), model.action_encoder(np.zeros((1, 3, 10))))
        return context
    return replay


def run_select(tmp_path, j, sources, manifest, target, *, compiler=compile_fixture, replay=None):
    model = helpers.Model()
    plan = s.planner(model=model, device=None, replay=replay or replay_fixture(model),
                     manifest_path=manifest["path"], manifest_sha256=manifest["sha256"],
                     target_path=target, audit_path=tmp_path / "audit.json",
                     fixture_compile=compiler, fixture_replay=True)
    result = online.online_select(j, B, t.FRAMES, sources, manifest, t.encoder, plan, tmp_path / "receipt.json")
    return result


def test_real_run_bank_fixture_to_durable_receipt_and_consumer(tmp_path):
    j, sources, manifest, target = setup(tmp_path)
    result = run_select(tmp_path, j, sources, manifest, target)
    receipt = json.loads((tmp_path / "receipt.json").read_bytes())
    audit_sha = receipt["plan"]["selected_id"].split("@")[1]
    assert receipt["plan"]["selected_id"].startswith("joint_h2_no_composition_sra_00@")
    audit = json.loads((tmp_path / "audit.json").read_bytes())
    assert audit["fixture"] and len(audit["search"]["attempts"]) == 40
    assert len(audit["search"]["ledger"]) == 84
    assert len(audit["model_entries"]) == 176 and len(audit["metered_events"]) == 176
    assert j.protocol.state(B)["attempted"]["predict_calls"] == 88
    assert j.protocol.state(B)["attempted"]["visual_encoder"] == 1
    j.call("stop", A, "fixture")
    j.call("stop", B, "fixture_no_physical")
    journal_sha = j.finish()
    out = s.consume_sra(j.path, journal_sha, [(tmp_path / "receipt.json", result["receipt_sha256"])],
                        tmp_path / "audit.json", audit_sha)
    assert out["fixture"] and out["new_model_calls"] == 0 and not out["equal_total_compute"]
    assert out["flops"] is None
    (tmp_path / "audit.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="audit hash"):
        s.consume_sra(j.path, journal_sha, [(tmp_path / "receipt.json", result["receipt_sha256"])],
                      tmp_path / "audit.json", audit_sha)


def test_budget_exhaustion_and_failures_no_selection(tmp_path):
    j, sources, manifest, target = setup(tmp_path)
    j.protocol.caps[B]["predict_calls"] = 5
    j.config["caps"][B]["predict_calls"] = 5
    with pytest.raises(ValueError):
        run_select(tmp_path, j, sources, manifest, target)
    assert j.protocol.state(B)["phase"] == "stopped"
    assert j.protocol.state(B)["attempted"]["predict_calls"] == 5
    assert not (tmp_path / "receipt.json").exists()
    j.close_partial()


def test_failed_replay_never_selects(tmp_path):
    j, sources, manifest, target = setup(tmp_path)
    def fail(context, actions):
        raise RuntimeError("fixture replay failure")
    with pytest.raises(ValueError, match="incomplete native"):
        run_select(tmp_path, j, sources, manifest, target, replay=fail)
    assert j.protocol.state(B)["phase"] == "stopped"
    assert not (tmp_path / "receipt.json").exists()
    assert not (tmp_path / "audit.json").exists()
    j.call("stop", A, "fixture")
    saved = t.jmod.consume(j.path, j.finish())
    assert saved["snapshot"]["rows"][1]["resources_attempted"]["predict_calls"] == 4


def test_saved_consumer_refuses_target_drift(tmp_path):
    j, sources, manifest, target = setup(tmp_path)
    result = run_select(tmp_path, j, sources, manifest, target)
    receipt = json.loads((tmp_path / "receipt.json").read_bytes())
    audit_sha = receipt["plan"]["selected_id"].split("@")[1]
    j.call("stop", A, "fixture")
    j.call("stop", B, "fixture")
    journal_sha = j.finish()
    target.write_bytes(b"drift")
    with pytest.raises(ValueError, match="source drift"):
        s.consume_sra(j.path, journal_sha, [(tmp_path / "receipt.json", result["receipt_sha256"])],
                      tmp_path / "audit.json", audit_sha)


def test_dependency_change_during_compile_is_refused(tmp_path):
    j, sources, manifest, target = setup(tmp_path)
    dep = tmp_path / "extra_dependency"
    dep.write_bytes(b"before")
    data = json.loads(Path(manifest["path"]).read_text())
    data["files"]["extra"] = {"path": str(dep), "sha256": s.file_sha(dep)}
    Path(manifest["path"]).write_text(json.dumps(data))
    manifest["sha256"] = s.file_sha(manifest["path"])
    def changing(model, device, ready, compiled, replay, emit):
        result = compile_fixture(model, device, ready, compiled, replay, emit)
        dep.write_bytes(b"after")
        return result
    with pytest.raises(ValueError, match="source inventory mismatch"):
        run_select(tmp_path, j, sources, manifest, target, compiler=changing)
    assert j.protocol.state(B)["phase"] == "stopped"
    assert not (tmp_path / "receipt.json").exists()
    j.call("stop", A, "fixture")
    j.finish()


@pytest.mark.parametrize("defect", ["target_drift", "manifest_drift", "missing_dep", "audit_exists"])
def test_source_or_audit_failure_closed(tmp_path, defect):
    j, sources, manifest, target = setup(tmp_path)
    if defect == "target_drift":
        target.write_bytes(b"changed")
    elif defect == "manifest_drift":
        Path(manifest["path"]).write_bytes(b"changed")
    elif defect == "missing_dep":
        value = json.loads(Path(manifest["path"]).read_text())
        del value["files"][s.REQUIRED[-1]]
        Path(manifest["path"]).write_text(json.dumps(value))
        manifest["sha256"] = s.file_sha(manifest["path"])
    else:
        (tmp_path / "audit.json").write_bytes(b"keep")
    with pytest.raises((ValueError, FileExistsError)):
        run_select(tmp_path, j, sources, manifest, target)
    assert not (tmp_path / "receipt.json").exists()
    if defect == "audit_exists":
        assert (tmp_path / "audit.json").read_bytes() == b"keep"
    j.close_partial()
