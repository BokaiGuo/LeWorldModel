import hashlib
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p4_feedback_closure_20261101_parallel as m


def write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    live = {
        "schema": "p4_feedback_live_cuda_20261027_v1.receipt",
        "status": "two_live_20step_3round_arms_completed",
        "P1_admission": False,
        "P4_admission": False,
        "independent_confirmation": False,
        "known_roots_only": True,
        "errors": [],
        "rows": [{"arm": "imagined", "status": "completed20"},
                 {"arm": "actual", "status": "completed20"}],
        "counts": {"encoder": {"attempted": 8, "completed": 8}},
        "model_hash_before": "a" * 64,
        "model_hash_after": "a" * 64,
    }
    audit = {
        "schema": "p4_feedback_live_saved_audit_20261027_v1.report",
        "status": "independent_saved_live_audit_passed",
        "P1_admission": False,
        "P4_admission": False,
        "independent_confirmation": False,
        "new_neural_calls": 0,
        "new_physical_steps": 0,
        "arms": [{"arm": "imagined"}, {"arm": "actual"}],
        "gates": {"source_model_goalQ_identity": True},
    }
    outcomes = {
        "schema": "p4_multiround_outcomes_20261027_v1",
        "status": "saved_arithmetic_audit_passed",
        "P1_admission": False,
        "P4_admission": False,
    }
    terminal = {
        "schema": "p4_terminal20_cuda_encoding_20261027_v1.derived_receipt",
        "status": "two_terminal20_latents_saved",
        "P1_admission": False,
        "P4_admission": False,
        "model_hash_before": "b" * 64,
        "model_hash_after": "b" * 64,
    }
    paths = {}
    for name, value in (("live", live), ("audit", audit), ("outcomes", outcomes),
                        ("terminal", terminal)):
        path = tmp_path / f"{name}.json"
        paths[name] = {"path": path, "sha256": write_json(path, value)}
    return paths


def test_saved_closure_is_engineering_only(tmp_path):
    report = m.build_report(fixture(tmp_path))
    assert report["status"] == "p4_engineering_saved_closure_verified"
    assert report["engineering"]["real_feedback_closed_loop_saved"] is True
    assert report["engineering"]["independent_saved_validation"] is True
    assert report["P4_admission"] is False
    assert report["independent_confirmation"] is False
    assert report["new_neural_calls"] == 0
    assert report["new_physical_steps"] == 0


@pytest.mark.parametrize("fault", ["sha", "admission", "arm", "model"])
def test_saved_closure_fails_closed(tmp_path, fault):
    paths = fixture(tmp_path)
    if fault == "sha":
        paths["live"]["sha256"] = "0" * 64
    else:
        target = paths["live"]["path"]
        data = json.loads(target.read_text())
        if fault == "admission":
            data["P4_admission"] = True
        elif fault == "arm":
            data["rows"][1]["arm"] = "imagined"
        else:
            data["model_hash_after"] = "c" * 64
        write_json(target, data)
        paths["live"]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        m.build_report(paths)


def test_write_new_refuses_overwrite(tmp_path):
    output = tmp_path / "p4_feedback_closure_20261101_parallel.json"
    m.write_new(output, {"ok": True})
    with pytest.raises(FileExistsError):
        m.write_new(output, {"ok": True})
