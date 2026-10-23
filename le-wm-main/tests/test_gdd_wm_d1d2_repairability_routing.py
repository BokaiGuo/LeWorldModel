import copy
import hashlib
import json

from experiments.gdd_wm.audit_d1d2_repairability_routing import audit


def _action(tag: str, index: int) -> list[list[float]]:
    # Distinct, deterministic action identities.  The exact values are not
    # task semantics; they let the audit test candidate provenance.
    digest = hashlib.sha256(f"{tag}:{index}".encode("utf-8")).digest()
    return [[(digest[0] - 128) / 256.0, (digest[1] - 128) / 256.0]]


def _package(
    *,
    leak: bool = False,
    outcome_trained: bool = False,
    held_candidates_overlap: bool = False,
) -> dict:
    signature = {
        "environment": "ToyMaze",
        "world_model_sha256": "wm-frozen-1",
        "horizon": 4,
        "candidate_family": "independent-context-v1",
        "action_frame": "raw_control",
        "task_cost_version": "toy-cost-v1",
    }
    metadata = {
        "teacher_model_id": "frozen_execution_supervised_control_v1",
        "teacher_model_frozen_before_d1": True,
        "teacher_fit_split": "train",
        "teacher_fit_root_ids": ["train_root"],
        "teacher_checkpoint_sha256": "a" * 64,
        "teacher_input_fields": [
            "source_latent",
            "predicted_terminal_latent",
            "goal_observation",
            "action_summary",
        ],
        "teacher_training_target_fields": [
            "real_task_cost" if outcome_trained else "observed_successor_latent",
        ],
        "teacher_training_uses_execution_outcomes": outcome_trained,
        "routing_signature": signature,
        "candidate_context_count_required": 3,
    }
    records = []
    for split, root in (("train", "train_root"), ("calibration", "cal_root"), ("outcome", "out_root")):
        root_tag = "train_root" if held_candidates_overlap and split == "outcome" else root
        for context in range(3):
            # a0/a1 are cross-context anchors; d{context} makes the candidate
            # set genuinely different in every context.
            for action_index, score, cost in ((0, 1.0, 0.2), (1, 0.0, 0.8), (context + 2, -0.5, 1.1)):
                value = -cost if leak else score
                records.append({
                    "instance_id": root,
                    "split": split,
                    "candidate_id": f"{root_tag}_candidate_{action_index}",
                    "teacher_set_id": f"{root}_context_{context}",
                    "candidate_bank_context_id": f"context_{context}",
                    "action_sequence": _action(root_tag, action_index),
                    "teacher_score": value,
                    "real_task_cost": cost,
                    "teacher_model_id": metadata["teacher_model_id"],
                    "teacher_score_source": "frozen_control_model_output",
                })
    return {"metadata": metadata, "records": records}


def _repairability(signature: dict, dataset: object) -> dict:
    return {
        "schema": "gdd_wm_finite_intervention_repairability.v0.1",
        "routing_signature": signature,
        "candidate_dataset_sha256": hashlib.sha256(
            json.dumps(dataset, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "probe_protocol": {
            "probe_count": 6,
            "fit_probe_count": 4,
            "holdout_probe_count": 2,
            "selection_uses_outcome": False,
            "heldout_effect_validation_passed": True,
            "local_support_passed": True,
            "mode_stability_passed": True,
            "margin_passed": True,
        },
    }


def test_permits_only_a_provenance_separated_teacher_and_matched_probe_gate(tmp_path) -> None:
    package = _package()
    dataset = tmp_path / "candidate_dataset.json"
    dataset.write_text(json.dumps(package, sort_keys=True), encoding="utf-8")
    evidence = tmp_path / "repairability.json"
    evidence.write_text(json.dumps(_repairability(package["metadata"]["routing_signature"], package)), encoding="utf-8")

    report = audit(dataset, evidence)

    assert report["status"] == "permitted_small_d2_smoke"
    assert report["teacher_control"]["passed"] is True
    assert report["repeated_context_stability"]["passed"] is True
    assert report["held_candidate_separation"]["passed"] is True
    assert report["finite_intervention_repairability"]["passed"] is True
    assert report["gpu_smoke"]["authorized"] is True


def test_rejects_exact_outcome_cost_teacher_even_when_contexts_are_stable(tmp_path) -> None:
    package = _package(leak=True)
    dataset = tmp_path / "candidate_dataset.json"
    dataset.write_text(json.dumps(package, sort_keys=True), encoding="utf-8")
    evidence = tmp_path / "repairability.json"
    evidence.write_text(json.dumps(_repairability(package["metadata"]["routing_signature"], package)), encoding="utf-8")

    report = audit(dataset, evidence)

    assert report["status"] == "rejected_fail_closed"
    assert report["teacher_control"]["exact_negative_cost_match"] is True
    assert "teacher_control" in report["rejection_reasons"]
    assert report["gpu_smoke"]["authorized"] is False


def test_rejects_teacher_declared_as_outcome_trained_even_if_scores_are_not_costs(tmp_path) -> None:
    package = _package(outcome_trained=True)
    dataset = tmp_path / "candidate_dataset.json"
    dataset.write_text(json.dumps(package, sort_keys=True), encoding="utf-8")
    evidence = tmp_path / "repairability.json"
    evidence.write_text(json.dumps(_repairability(package["metadata"]["routing_signature"], package)), encoding="utf-8")

    report = audit(dataset, evidence)

    assert report["status"] == "rejected_fail_closed"
    assert report["teacher_control"]["forbidden_target_fields"] == ["real_task_cost"]
    assert report["teacher_control"]["metadata_checks"]["teacher_training_outcome_free"] is False
    assert "teacher_control" in report["rejection_reasons"]


def test_rejects_reused_training_candidates_in_the_held_candidate_bank(tmp_path) -> None:
    package = _package(held_candidates_overlap=True)
    dataset = tmp_path / "candidate_dataset.json"
    dataset.write_text(json.dumps(package, sort_keys=True), encoding="utf-8")
    evidence = tmp_path / "repairability.json"
    evidence.write_text(json.dumps(_repairability(package["metadata"]["routing_signature"], package)), encoding="utf-8")

    report = audit(dataset, evidence)

    assert report["status"] == "rejected_fail_closed"
    assert report["held_candidate_separation"]["held_novel_fraction"] == 0.0
    assert "held_candidate_separation" in report["rejection_reasons"]


def test_rejects_repairability_evidence_from_a_different_decision_cell(tmp_path) -> None:
    package = _package()
    dataset = tmp_path / "candidate_dataset.json"
    dataset.write_text(json.dumps(package, sort_keys=True), encoding="utf-8")
    signature = copy.deepcopy(package["metadata"]["routing_signature"])
    signature["environment"] = "OtherEnv"
    evidence = tmp_path / "repairability.json"
    evidence.write_text(json.dumps(_repairability(signature, package)), encoding="utf-8")

    report = audit(dataset, evidence)

    assert report["status"] == "rejected_fail_closed"
    assert report["finite_intervention_repairability"]["signature_matches"] is False
    assert "finite_intervention_repairability" in report["rejection_reasons"]
