import json

from experiments.gdd_wm.audit_preference_stability import audit_records
from experiments.gdd_wm.build_repeated_candidate_contexts import main as build_repeated_contexts_main


def _row(root, set_id, candidate, score, cost, realized="real://x"):
    return {
        "instance_id": root,
        "teacher_set_id": set_id,
        "candidate_id": candidate,
        "teacher_score": score,
        "real_task_cost": cost,
        "split": "outcome",
        "teacher_model_id": "test_teacher",
        "candidate_generator": "test_bank",
        "predicted_latent_ref": "pred://x",
        "realized_latent_ref": realized,
    }


def test_stability_requires_repeated_candidate_set_contexts():
    report = audit_records([
        _row("r0", "s0", 0, 2.0, 0.1),
        _row("r0", "s0", 1, 1.0, 0.3),
    ])
    assert report["status"] == "rejected_fail_closed"
    assert report["preference_stability"]["status"] == "unavailable_insufficient_candidate_set_repeats"
    assert report["execution_consistency"]["all"]["execution_consistency"] == 1.0
    assert report["teacher_execution_leakage"]["exact_negative_cost_match"] is False


def test_repeated_contexts_can_make_stability_estimable():
    rows = [
        _row("r0", "s0", 0, 2.0, 0.1),
        _row("r0", "s0", 1, 1.0, 0.3),
        _row("r0", "s1", 0, 2.0, 0.1),
        _row("r0", "s1", 1, 1.0, 0.3),
    ]
    report = audit_records(rows)
    assert report["preference_stability"]["status"] == "computed"
    assert report["status"] == "passed"


def test_execution_derived_teacher_is_rejected_as_leakage():
    rows = [
        _row("r0", "s0", 0, -0.1, 0.1),
        _row("r0", "s0", 1, -0.3, 0.3),
        _row("r0", "s1", 0, -0.1, 0.1),
        _row("r0", "s1", 1, -0.3, 0.3),
    ]
    report = audit_records(rows)
    assert report["teacher_execution_leakage"]["exact_negative_cost_match"] is True
    assert report["teacher_execution_leakage"]["independent_execution_consistency_valid"] is False
    assert report["status"] == "rejected_fail_closed"


def test_repeated_context_builder_matches_actions_not_candidate_indices(tmp_path, monkeypatch):
    common = {
        "instance_id": "r0",
        "split": "outcome",
        "environment": "toy",
        "teacher_model_id": "external_teacher",
        "teacher_score": 1.0,
        "real_task_cost": 0.5,
        "predicted_latent_ref": "pred://x",
        "realized_latent_ref": "real://x",
    }
    first = dict(common, candidate_id=0, action_sequence=[[0.1, 0.2]])
    second = dict(common, candidate_id=7, action_sequence=[[0.1, 0.2]])
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    output = tmp_path / "out.json"
    left.write_text(json.dumps({"metadata": {"candidate_bank_hash": "left"}, "records": [first]}))
    right.write_text(json.dumps({"metadata": {"candidate_bank_hash": "right"}, "records": [second]}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_repeated_candidate_contexts",
            "--input", str(left),
            "--input", str(right),
            "--output", str(output),
        ],
    )
    build_repeated_contexts_main()
    payload = json.loads(output.read_text())
    assert payload["metadata"]["matched_root_action_pairs"] == 1
    assert payload["metadata"]["context_count"] == 2
    assert len({row["candidate_id"] for row in payload["records"]}) == 1
    assert len({row["teacher_set_id"] for row in payload["records"]}) == 2
