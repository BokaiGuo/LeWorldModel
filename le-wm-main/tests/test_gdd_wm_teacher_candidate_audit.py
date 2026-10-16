import json

from experiments.gdd_wm.audit_teacher_candidate_packages import audit_package


def _row(teacher_model="external_teacher_v1", leaked=False, context="ctx0", realized="latent://x"):
    real = 0.5
    return {
        "instance_id": "root0",
        "candidate_id": "a",
        "matched_action_hash": "action_a",
        "teacher_set_id": f"root0__{context}",
        "candidate_bank_context_id": context,
        "teacher_model_id": teacher_model,
        "teacher_score": -real if leaked else 1.0,
        "real_task_cost": real,
        "predicted_latent_ref": "pred://x",
        "realized_latent_ref": realized,
        "split": "outcome",
        "candidate_generator": "fixed",
    }


def test_teacher_audit_rejects_leakage_and_missing_runtime_refs(tmp_path):
    path = tmp_path / "candidate_dataset.json"
    path.write_text(json.dumps({"metadata": {}, "records": [_row(leaked=True, realized=None)]}))
    report = audit_package(path)
    assert report["eligible_for_d1_and_d2"] is False
    assert report["teacher_execution_leakage"]["exact_negative_cost_match"] is True
    assert report["gates"]["complete_outcome_runtime_refs"] is False


def test_teacher_audit_accepts_only_repeated_external_contexts(tmp_path):
    path = tmp_path / "candidate_dataset.json"
    path.write_text(json.dumps({"metadata": {}, "records": [_row(context="ctx0"), _row(context="ctx1")]}))
    report = audit_package(path)
    assert report["eligible_for_d1_and_d2"] is True
    assert report["summary"]["repeated_root_candidate_pairs"] == 1
