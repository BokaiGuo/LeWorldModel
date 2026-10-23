from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.gdd_wm.validate_pointmaze_counterfactual_endpoint_v6_panel import validate


RESUME = Path(__file__).resolve().parents[1] / "experiments" / "gdd_wm" / "resume_pointmaze_counterfactual_endpoint_v6_cuda.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_valid_panel(tmp_path: Path) -> tuple[Path, Path]:
    panel = tmp_path / "panel"
    panel.mkdir()
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({
        "schema": "pointmaze_counterfactual_endpoint_v6_static_preflight.v1",
        "status": "passed_static_preflight",
        "root_counts": {"train": 1, "calibration": 1, "outcome": 1},
    }), encoding="utf-8")
    context = panel / "temporal_ood_v6_context.json"
    context.write_text(json.dumps({
        "metadata": {"schema": "pointmaze_temporal_ood_v6_matched_panel.v3", "frozen_preflight_ledger": {
            "schema": "pointmaze_counterfactual_endpoint_v6_static_preflight.v1",
            "status": "passed_static_preflight", "sha256": _sha256(preflight),
            "root_counts": {"train": 1, "calibration": 1, "outcome": 1}, "path": str(preflight.resolve()),
        }},
        "records": [
            {"instance_id": "train", "split": "train"},
            {"instance_id": "cal", "split": "calibration"},
            {"instance_id": "out", "split": "outcome"},
        ],
    }), encoding="utf-8")
    source = panel / "temporal_ood_v6_source_dataset.json"
    source.write_text(json.dumps({"records": [{"instance_id": "train", "split": "train"}]}), encoding="utf-8")
    arrays = panel / "temporal_ood_v6_source_arrays.npz"
    arrays.write_bytes(b"synthetic arrays")
    calibration = panel / "temporal_ood_v6_calibration_outcomes.json"
    calibration.write_text(json.dumps({"cal/0": {"realized_terminal_state": [0.0, 0.0, 0.0, 0.0]}}), encoding="utf-8")
    artifacts = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in {
            "context": context, "source_dataset": source, "source_arrays": arrays, "calibration_outcomes": calibration,
        }.items()
    }
    artifacts["outcome_labels"] = {"materialized": False, "path": str(panel / "temporal_ood_v6_outcome_labels.json"), "sha256": None}
    (panel / "temporal_ood_v6_manifest.json").write_text(json.dumps({"artifacts": artifacts}), encoding="utf-8")
    return panel, preflight


def test_resume_validator_binds_artifacts_and_sealed_root_roles(tmp_path: Path) -> None:
    panel, preflight = _write_valid_panel(tmp_path)
    result = validate(panel=panel, frozen_preflight=preflight)
    assert result["status"] == "passed_immutable_v6_resume_panel_validation"
    assert result["outcome_labels_sealed"] is True
    assert result["root_counts"] == {"train": 1, "calibration": 1, "outcome": 1}


def test_resume_validator_rejects_outcome_label_leakage(tmp_path: Path) -> None:
    panel, preflight = _write_valid_panel(tmp_path)
    context = panel / "temporal_ood_v6_context.json"
    payload = json.loads(context.read_text(encoding="utf-8"))
    payload["records"][-1]["real_task_cost"] = 0.0
    context.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash/path mismatch"):
        validate(panel=panel, frozen_preflight=preflight)


def test_resume_launcher_never_rebuilds_or_overwrites_a_completed_panel() -> None:
    text = RESUME.read_text(encoding="utf-8")
    assert 'PANEL_VALIDATOR="experiments.gdd_wm.validate_pointmaze_counterfactual_endpoint_v6_panel"' in text
    assert 'missing completed immutable V6 panel' in text
    assert 'refusing to overwrite V6 endpoint-control output' in text
    assert 'pointmaze_temporal_ood_v6_matched_panel' not in text
    assert 'preflight_pointmaze_counterfactual_endpoint_v6' not in text
    assert '--ensemble-size 4' in text
    assert '--minimum-summary-collision-pairs-per-root 8' in text
