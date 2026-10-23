import json

import pytest

from experiments.decision_benchmark.recover_pointmaze_post_execution_labels import run


def test_recovery_refuses_uncompleted_or_already_serialized_reports(tmp_path) -> None:
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"records": []}), encoding="utf-8")
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"status": "incomplete"}), encoding="utf-8")
    with pytest.raises(ValueError, match="completed repairability"):
        run(context=context, completed_report=incomplete, output=tmp_path / "labels.json")

    complete = tmp_path / "complete.json"
    complete.write_text(json.dumps({
        "status": "completed_execution_grounded_repairability_control",
        "post_execution_outcome_labels": {},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="already embeds"):
        run(context=context, completed_report=complete, output=tmp_path / "labels2.json")
