import json

from experiments.gdd_wm.audit_frozen_teacher_failure_modes import audit


def test_failure_mode_audit_is_root_stratified(tmp_path):
    rows = []
    for root, costs in (("r0", (0.1, 0.4)), ("r1", (0.4, 0.1))):
        for index, cost in enumerate(costs):
            rows.append({
                # The second root deliberately inverts execution relative
                # to the frozen teacher score.
                "teacher_score": (-0.1, -0.4)[index],
                "real_task_cost": cost,
                "action_support": {"min": -0.5, "max": 0.5},
                "split": "train",
                "candidate_bank_context_id": "ctx",
                "predicted_horizon": 4,
                "start_episode_id": root,
                "instance_id": root,
                "teacher_set_id": root,
                "candidate_id": str(index),
            })
    path = tmp_path / "candidate_dataset.json"
    path.write_text(json.dumps({"records": rows}), encoding="utf-8")
    report = audit(path)
    assert report["global"]["pair_count"] == 2
    assert report["strata"]["root"]["r0"]["execution_consistency"] == 1.0
    assert report["strata"]["root"]["r1"]["execution_consistency"] == 0.0
    assert report["protocol"]["planner_authorization"] is False
