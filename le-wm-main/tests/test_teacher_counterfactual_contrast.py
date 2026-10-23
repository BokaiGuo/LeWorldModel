import json

import numpy as np

from experiments.gdd_wm.audit_teacher_counterfactual_contrast import audit


def _dataset(tmp_path, *, predicted_xy, realized_xy, costs, scores, latent_values=None):
    source = tmp_path / "source"
    source.mkdir()
    arrays = {}
    records = []
    latent_values = predicted_xy if latent_values is None else latent_values
    for index, (predicted, realized, latent, cost, score) in enumerate(zip(predicted_xy, realized_xy, latent_values, costs, scores)):
        key = f"candidate/{index}"
        arrays[f"predicted::{key}"] = np.asarray(latent, dtype=np.float64)
        arrays[f"realized::{key}"] = np.asarray([*realized, 0.0, 0.0], dtype=np.float64)
        records.append({
            "instance_id": "root-1", "teacher_set_id": "root-1/context-1", "candidate_id": str(index),
            "split": "outcome", "teacher_score": float(score), "real_task_cost": float(cost),
            "teacher_predicted_terminal_xy": list(predicted),
            "predicted_terminal_latent_ref": f"arrays.npz::predicted::{key}",
            "realized_terminal_state_ref": f"arrays.npz::realized::{key}",
            "predicted_latent_source_dataset": str(source / "candidate_dataset.json"),
        })
    np.savez(source / "arrays.npz", **arrays)
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"metadata": {"teacher_model_id": "test"}, "records": records}), encoding="utf-8")
    return dataset


def test_audit_identifies_predicted_action_effect_collapse(tmp_path):
    dataset = _dataset(
        tmp_path,
        predicted_xy=[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        realized_xy=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        costs=[3.0, 2.0, 1.0], scores=[3.0, 2.0, 1.0],
    )
    report = audit(dataset)
    row = report["per_decision_group"][0]
    assert row["dominant_failure_mode"] == "predicted_action_effect_collapse"
    assert row["predicted_to_realized_spread_ratio"] == 0.0
    assert report["protocol"]["policy_selection_or_authorization"] is False


def test_audit_identifies_dynamics_mismatch_before_readout_mismatch(tmp_path):
    dataset = _dataset(
        tmp_path,
        predicted_xy=[(0.0, 0.0), (1.0, 0.0), (3.0, 0.0)],
        realized_xy=[(0.0, 0.0), (0.0, 3.0), (1.0, 0.0)],
        costs=[3.0, 2.0, 1.0], scores=[3.0, 2.0, 1.0],
    )
    report = audit(dataset)
    row = report["per_decision_group"][0]
    assert row["predicted_to_realized_spread_ratio"] >= 0.25
    assert row["predicted_vs_realized_endpoint_pairwise_correlation"] < 0.30
    assert row["dominant_failure_mode"] == "predicted_dynamics_mismatch"


def test_audit_identifies_task_readout_mismatch_after_geometry_agrees(tmp_path):
    dataset = _dataset(
        tmp_path,
        predicted_xy=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        realized_xy=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        costs=[3.0, 2.0, 1.0], scores=[3.0, 2.0, 1.0],
    )
    report = audit(dataset)
    row = report["per_decision_group"][0]
    assert row["predicted_vs_realized_endpoint_pairwise_correlation"] == 1.0
    assert row["teacher_task_pair_agreement"] == 0.0
    assert row["dominant_failure_mode"] == "task_readout_mismatch"


def test_audit_separates_endpoint_readout_geometry_from_dynamics_mismatch(tmp_path):
    realized = [(0.0, 0.0), (0.0, 3.0), (1.0, 0.0)]
    dataset = _dataset(
        tmp_path,
        predicted_xy=[(0.0, 0.0), (1.0, 0.0), (3.0, 0.0)],
        realized_xy=realized,
        latent_values=realized,
        costs=[3.0, 2.0, 1.0], scores=[3.0, 2.0, 1.0],
    )
    row = audit(dataset)["per_decision_group"][0]
    assert row["predicted_latent_vs_realized_endpoint_pairwise_correlation"] == 1.0
    assert row["predicted_vs_realized_endpoint_pairwise_correlation"] < 0.30
    assert row["dominant_failure_mode"] == "endpoint_readout_geometry_mismatch"


def test_audit_uses_only_declared_shared_anchor_provenance_for_realized_state(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    context = tmp_path / "shared" / "contexts" / "context_00"
    context.mkdir(parents=True)
    keys = {}
    records = []
    for index in range(3):
        key = f"root/{index}"
        keys[f"predicted::{key}"] = np.asarray([float(index), 0.0, 0.0])
        keys[f"realized::{key}"] = np.asarray([float(index), 0.0, 0.0, 0.0])
        records.append({
            "instance_id": "root", "teacher_set_id": "root/context", "candidate_id": str(index),
            "split": "outcome", "teacher_score": float(2 - index), "real_task_cost": float(index),
            "teacher_predicted_terminal_xy": [float(index), 0.0],
            "predicted_terminal_latent_ref": f"arrays.npz::predicted::{key}",
            "realized_terminal_state_ref": f"arrays.npz::realized::{key}",
            "predicted_latent_source_dataset": str(context / "candidate_dataset.json"),
        })
    np.savez(context / "arrays.npz", **{key: value for key, value in keys.items() if key.startswith("predicted::")})
    np.savez(source / "arrays.npz", **{key: value for key, value in keys.items() if key.startswith("realized::")})
    (tmp_path / "shared" / "protocol_report.json").write_text(
        json.dumps({"metadata": {}, "records": [], "source_candidate_dataset": str(source / "candidate_dataset.json")}),
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"metadata": {"teacher_model_id": "test"}, "records": records}), encoding="utf-8")
    assert audit(dataset)["per_decision_group"][0]["dominant_failure_mode"] == "no_dominant_failure_identified"
