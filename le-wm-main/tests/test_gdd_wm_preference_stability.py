from experiments.gdd_wm.audit_preference_stability import audit_records


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
