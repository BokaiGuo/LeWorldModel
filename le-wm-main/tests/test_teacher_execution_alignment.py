import json

from experiments.gdd_wm.calibrate_teacher_execution_alignment import audit


def _dataset(*, correct: bool, roots: int = 20) -> dict:
    records = []
    for index in range(roots):
        root = f"cal-{index}"
        # Higher teacher score prefers candidate a.  Swap realized costs to
        # construct the independent execution-inversion control.
        costs = (0.1, 0.9) if correct else (0.9, 0.1)
        for candidate, score, cost in (("a", 1.0, costs[0]), ("b", 0.0, costs[1])):
            records.append({
                "instance_id": root, "teacher_set_id": f"{root}-set", "candidate_id": candidate,
                "teacher_score": score, "real_task_cost": cost, "teacher_model_id": "teacher-v1",
                "split": "calibration",
            })
        # Outcome rows deliberately contradict the calibration rows.  The
        # audit must not read them or change its gate decision.
        records.append({
            "instance_id": f"out-{index}", "teacher_set_id": f"out-{index}-set", "candidate_id": "a",
            "teacher_score": 1.0, "real_task_cost": 0.9, "teacher_model_id": "teacher-v1", "split": "outcome",
        })
    return {"metadata": {"teacher_model_id": "teacher-v1"}, "records": records}


def test_calibration_alignment_passes_for_teacher_that_orders_every_calibration_root(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(_dataset(correct=True)), encoding="utf-8")
    result = audit(path, draws=400, seed=3)
    assert result["status"] == "passed_calibration_alignment"
    assert result["outcome_rows_read"] is False
    assert result["calibration"]["root_bootstrap_lower_bound"] == 1.0


def test_calibration_alignment_rejects_execution_inverted_teacher(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(_dataset(correct=False)), encoding="utf-8")
    result = audit(path, draws=400, seed=3)
    assert result["status"] == "rejected_calibration_alignment"
    assert result["calibration"]["root_bootstrap_lower_bound"] == 0.0


def test_calibration_alignment_refuses_insufficient_root_support(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(_dataset(correct=True, roots=3)), encoding="utf-8")
    result = audit(path, draws=400, seed=3, minimum_roots=20)
    assert result["status"] == "rejected_insufficient_calibration_roots"
