import json

from experiments.gdd_wm.audit_cross_root_action_effect_consistency import audit


def test_cross_root_audit_detects_shared_bank_rank_flip(tmp_path):
    records = []
    for root, costs in (("r0", (0.1, 0.9)), ("r1", (0.8, 0.2))):
        for candidate, cost in enumerate(costs):
            records.append(
                {
                    "instance_id": root,
                    "candidate_id": candidate,
                    "split": "outcome",
                    "action_sequence": [[candidate, 0.0]],
                    "real_task_cost": cost,
                }
            )
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"metadata": {"shared_candidate_bank": True}, "records": records}))
    report = audit(dataset)
    outcome = report["splits"]["outcome"]
    assert outcome["shared_action_bank"] is True
    assert outcome["pair_flip_fraction"] == 1.0
    assert outcome["mean_majority_consistency"] == 0.5


def test_cross_root_audit_canonicalizes_goal_relative_bank(tmp_path):
    records = []
    goal = [1.0, 0.0]
    for root, source, actions in (
        ("r0", [0.0, 0.0, 0.0, 0.0], [[[1.0, 0.0]]]),
        ("r1", [0.0, 1.0, 0.0, 0.0], [[[2**-0.5, -2**-0.5]]]),
    ):
        records.append(
            {
                "instance_id": root,
                "candidate_id": 0,
                "split": "outcome",
                "source_state_raw": source,
                "goal_raw": goal,
                "action_sequence": actions,
                "real_task_cost": 0.1,
            }
        )
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "metadata": {"candidate_bank_coordinate_frame": "goal_relative"},
                "records": records,
            }
        )
    )
    report = audit(dataset)
    assert report["splits"]["outcome"]["shared_action_bank"] is True
