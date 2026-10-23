import hashlib
import json

import numpy as np

from experiments.gdd_wm.audit_prospective_repairability_routing import run


def _fold(root: str) -> int:
    return int(hashlib.sha256(root.encode("utf-8")).hexdigest()[:8], 16) % 2


def _roots(split: str, count: int, required_fold=None):
    values = []
    index = 0
    while len(values) < count:
        root = f"{split}-{index}"
        if required_fold is None or _fold(root) == required_fold:
            values.append(root)
        index += 1
    return values


def _payload(tmp_path):
    records, audit_rows = [], []
    modes = [
        "endpoint_readout_geometry_mismatch",
        "task_readout_mismatch",
        "predicted_dynamics_mismatch",
        "predicted_action_effect_collapse",
        "no_dominant_failure_identified",
    ]
    split_roots = {
        "train": _roots("train", 9),
        "calibration": _roots("cal", 12),
        "outcome": _roots("out", 9),
    }
    for split, roots in split_roots.items():
        for root_index, root in enumerate(roots):
            mode = modes[root_index % len(modes)]
            context = f"{root}/ctx"
            feature = modes.index(mode) * 10.0
            for candidate in range(3):
                records.append({
                    "instance_id": root, "teacher_set_id": context, "candidate_id": str(candidate), "split": split,
                    "teacher_predicted_terminal_xy": [feature + candidate, 0.0],
                    "action_sequence": [[0.1 * candidate, 0.0]] * 20,
                    "teacher_score": float(2 - candidate),
                    "source_latent_model": [feature, 1.0],
                    "source_history_model": [feature, 0.0, feature, 1.0],
                    "goal_raw": [1.0, 1.0],
                })
            audit_rows.append({"instance_id": root, "teacher_set_id": context, "dominant_failure_mode": mode})
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"metadata": {}, "records": records}), encoding="utf-8")
    audit = tmp_path / "contrast.json"
    audit.write_text(json.dumps({
        "schema": "gdd_wm_teacher_counterfactual_contrast_audit.v0.1",
        "protocol": {"dataset": str(dataset.resolve()), "candidate_dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()},
        "per_decision_group": audit_rows,
    }), encoding="utf-8")
    return dataset, audit


def test_prospective_router_keeps_outcome_labels_out_of_fit_and_authorizes_separable_routes(tmp_path):
    dataset, contrast = _payload(tmp_path)
    result = run(dataset, contrast, seed=7, minimum_mode_count=1)
    assert result["status"] == "authorized_prospective_mechanism_router"
    assert result["protocol"]["outcome_labels_read_before_policy_freeze"] is False
    assert result["outcome"]["deployed"]["accuracy"] == 1.0
    assert result["outcome"]["deployed"]["mean_repair_utility"] > 0.0
    assert result["train_feature_support"]["outcome_coverage"] == 1.0


def test_prospective_router_rejects_when_train_feature_support_does_not_transport(tmp_path):
    dataset, contrast = _payload(tmp_path)
    payload = json.loads(dataset.read_text())
    for row in payload["records"]:
        if row["split"] == "calibration":
            row["teacher_predicted_terminal_xy"] = [1000.0, 1000.0]
    dataset.write_text(json.dumps(payload), encoding="utf-8")
    contrast_payload = json.loads(contrast.read_text())
    contrast_payload["protocol"]["candidate_dataset_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    contrast.write_text(json.dumps(contrast_payload), encoding="utf-8")
    result = run(dataset, contrast, seed=7, minimum_mode_count=1)
    assert result["status"] == "rejected_train_feature_support_transport"
    assert result["train_feature_support"]["transport_supported"] is False


def test_prospective_router_quarantines_family_without_calibration_submechanism_support(tmp_path):
    dataset, contrast = _payload(tmp_path)
    payload = json.loads(contrast.read_text())
    for row in payload["per_decision_group"]:
        if row["instance_id"].startswith("cal-") and row["dominant_failure_mode"] == "endpoint_readout_geometry_mismatch":
            row["dominant_failure_mode"] = "task_readout_mismatch"
    contrast.write_text(json.dumps(payload), encoding="utf-8")
    result = run(dataset, contrast, seed=7, minimum_mode_count=1)
    assert "metric_repair" not in result["calibration"]["allowed_repair_families"]
    assert result["calibration"]["repair_family_support"]["metric_repair"]["supported"] is False


def test_prospective_router_refuses_mismatched_audit_provenance(tmp_path):
    dataset, contrast = _payload(tmp_path)
    payload = json.loads(contrast.read_text())
    payload["protocol"]["candidate_dataset_sha256"] = "wrong"
    contrast.write_text(json.dumps(payload), encoding="utf-8")
    try:
        run(dataset, contrast)
    except ValueError as error:
        assert "SHA256" in str(error)
    else:
        raise AssertionError("expected provenance mismatch to fail closed")
