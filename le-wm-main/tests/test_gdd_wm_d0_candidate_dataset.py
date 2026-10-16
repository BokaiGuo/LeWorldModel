import json

from experiments.gdd_wm.build_shared_candidate_dataset import build_artifact, validate_records


PROVENANCE = {
    "outcome_used_for_teacher_or_metric_selection": False,
    "snapshot_selection_from_outcome": False,
    "teacher_frozen_before_distillation": True,
    "shared_candidate_bank": True,
    "runtime_teacher_forbidden": True,
}


def _row(split="train", root=1, candidate=0):
    return {
        "instance_id": "instance-1",
        "split": split,
        "environment": "toy",
        "start_episode_id": root,
        "goal_id": "goal-1",
        "candidate_id": candidate,
        "action_sequence": [[0.0, 0.1]],
        "candidate_generator": "random",
        "cem_stage": "initial",
        "source_observation_ref": "obs:source",
        "goal_observation_ref": "obs:goal",
        "teacher_model_id": "teacher-pending",
        "teacher_score": float(candidate),
        "teacher_rank": candidate + 1,
        "teacher_set_id": "set-1",
        "real_success": None,
        "real_task_cost": None,
        "realized_observation_ref": None,
        "predicted_latent_ref": None,
        "realized_latent_ref": None,
        "action_support": {"source": "toy", "in_support": True},
        "privileged_state_role": "offline_label_only",
    }


def test_d0_rejects_missing_provenance_and_incomplete_sets():
    report = validate_records([_row(candidate=0)], {}, expected_candidates=2)
    assert report["status"] == "rejected_fail_closed"
    assert report["decision"]["d0_schema_complete"] is False
    assert any("provenance metadata missing" in error for error in report["errors"])


def test_d0_canonicalizes_valid_schema_but_keeps_scientific_gate_pending(tmp_path):
    rows = [_row(candidate=0), _row(candidate=1)]
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"metadata": PROVENANCE, "records": rows}))
    report = build_artifact(source, tmp_path / "out", expected_candidates=2)
    assert report["status"] == "passed"
    assert report["decision"]["d0_schema_complete"] is True
    assert report["decision"]["teacher_execution_validation"] == "pending"
    assert report["decision"]["scientific_readiness"] == "blocked_teacher_execution_validation"
    assert (tmp_path / "out" / "candidate_records.jsonl").exists()


def test_d0_rejects_root_leakage_and_missing_action_support():
    rows = [_row(split="train", root=7, candidate=0), _row(split="outcome", root=7, candidate=1)]
    rows[1]["action_support"] = None
    report = validate_records(rows, PROVENANCE, expected_candidates=1)
    assert report["status"] == "rejected_fail_closed"
    assert any("root-disjoint violation" in error for error in report["errors"])
    assert any("action_support is missing" in error for error in report["errors"])


def test_d0_surfaces_exact_teacher_execution_leakage():
    rows = [_row(candidate=0), _row(candidate=1)]
    for index, row in enumerate(rows):
        row["real_task_cost"] = float(index + 1)
        row["teacher_score"] = -row["real_task_cost"]
    report = validate_records(rows, PROVENANCE, expected_candidates=2)
    assert report["status"] == "passed"
    assert report["summary"]["teacher_score_exact_negative_real_task_cost"] is True
    assert report["decision"]["teacher_execution_validation"] == "rejected_exact_negative_cost_leakage"
    assert report["decision"]["d1_ready"] is False
    assert any("label leakage" in warning for warning in report["warnings"])
