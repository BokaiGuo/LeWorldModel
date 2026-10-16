import json

from experiments.gdd_wm.summarize_train_only_teacher_feature_panel import summarize


def _write(path, offset):
    records = []
    for candidate, (score, cost) in enumerate(((1.0 + offset, 0.1), (0.0 + offset, 0.8))):
        records.append({
            "instance_id": "heldout_root",
            "candidate_id": str(candidate),
            "split": "outcome",
            "teacher_score": score,
            "real_task_cost": cost,
            "real_success": candidate == 0,
        })
    path.write_text(json.dumps({"records": records}))


def test_feature_panel_uses_common_root_bootstrap(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write(first, 0.0)
    _write(second, 0.1)
    report = summarize({"predicted_terminal": first, "action_summary": second}, draws=100, seed=1)
    assert report["status"] == "completed_diagnostic"
    assert report["families"]["predicted_terminal"]["outcome_root_count"] == 1
    assert report["paired_deltas"]["action_summary"]["vs_predicted_terminal"]["regret_delta"]["n"] == 1
