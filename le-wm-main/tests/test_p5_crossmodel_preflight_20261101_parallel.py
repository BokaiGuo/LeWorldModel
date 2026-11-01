import hashlib
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p5_crossmodel_preflight_20261101_parallel as m


def write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    panel = {
        "schema": "p5_crossmodel_panel_binding_20261030_codex_v1",
        "status": "saved_input_physical_identity_bound_not_scientific_admission",
        "P1_admission": False,
        "P5_admission": False,
        "branch_denominator": 164,
        "pair_denominator": 80,
        "known_roots": 2,
        "fresh_roots_certified": 0,
        "permissions": {
            "jepa": "history RGB plus agent position and velocity",
            "lewm": "RGB-only; model task binding inherited from physical receipt",
        },
    }
    abi = {
        "schema": "p5_jepa_rgb_checkpoint_abi_20261030_codex_v1",
        "status": "static_native_abi_and_rgb_only_gap_recorded",
        "P1_admission": False,
        "P5_admission": False,
        "pinned_sha256": {"checkpoint": "9" * 64},
        "model_config_from_pinned_loader": {"use_proprio": True},
        "native_h1_predictor_input_abi": {
            "visual": {}, "action": {}, "proprio": {}
        },
        "analysis": {"same_permission_status": "blocked: fixture"},
    }
    paths = {}
    for name, value in (("panel", panel), ("jepa_abi", abi)):
        path = tmp_path / f"{name}.json"
        paths[name] = {"path": path, "sha256": write_json(path, value)}
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    paths["lewm_checkpoint"] = {
        "path": checkpoint, "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    }
    return paths


def test_prepare_preserves_permission_and_independence_blockers(tmp_path):
    report = m.build_prepare(fixture(tmp_path))
    assert report["status"] == "p5_crossmodel_cuda_preflight_prepared"
    assert report["same_permission_comparison_ready"] is False
    assert report["fresh_roots_certified"] == 0
    assert report["P5_admission"] is False
    assert report["independent_confirmation"] is False
    assert "native_jepa_requires_proprio" in report["blockers"]


@pytest.mark.parametrize("fault", ["sha", "permission", "fresh", "admission"])
def test_prepare_fails_closed(tmp_path, fault):
    paths = fixture(tmp_path)
    if fault == "sha":
        paths["panel"]["sha256"] = "0" * 64
    else:
        target = paths["panel"]["path"] if fault != "permission" else paths["jepa_abi"]["path"]
        data = json.loads(target.read_text())
        if fault == "permission":
            del data["native_h1_predictor_input_abi"]["proprio"]
        elif fault == "fresh":
            data["fresh_roots_certified"] = 1
        else:
            data["P5_admission"] = True
        paths["panel" if target == paths["panel"]["path"] else "jepa_abi"]["sha256"] = write_json(target, data)
    with pytest.raises(ValueError):
        m.build_prepare(paths)


def test_cuda_receipt_requires_lock_cuda_and_frozen_hash():
    prepared = {"P5_admission": False, "independent_confirmation": False}
    with pytest.raises(ValueError):
        m.build_cuda_receipt(prepared, {"lock_held": False, "cuda_available": True,
                                        "checkpoint_before": "a", "checkpoint_after": "a"})
    with pytest.raises(ValueError):
        m.build_cuda_receipt(prepared, {"lock_held": True, "cuda_available": False,
                                        "checkpoint_before": "a", "checkpoint_after": "a"})
    with pytest.raises(ValueError):
        m.build_cuda_receipt(prepared, {"lock_held": True, "cuda_available": True,
                                        "checkpoint_before": "a", "checkpoint_after": "b"})


def test_verify_reconstructs_saved_receipt(tmp_path):
    prepare = m.build_prepare(fixture(tmp_path))
    cuda = m.build_cuda_receipt(
        prepare,
        {"lock_held": True, "cuda_available": True, "device_index": 0,
         "device_name": "fixture", "checkpoint_before": prepare["lewm_checkpoint_sha256"],
         "checkpoint_after": prepare["lewm_checkpoint_sha256"], "strict_load": True,
         "frozen": True, "eval": True, "neural_forward_calls": 0},
    )
    report = m.verify_saved(prepare, cuda)
    assert report["status"] == "p5_crossmodel_cuda_preflight_saved_validation_passed"
    assert report["P5_admission"] is False
    assert report["same_permission_comparison_ready"] is False
