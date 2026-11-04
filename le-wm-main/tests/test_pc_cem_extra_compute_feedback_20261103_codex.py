import json
from pathlib import Path

from experiments import pc_cem_extra_compute_feedback_20261103_codex as run


def test_saved_panel_tokens_and_action_bindings():
    sha = run.old.prior.sha
    physical_path = run.PREFIX.with_name(run.PREFIX.name + "_physical.json")
    physical = json.loads(physical_path.read_text())
    assert len(physical["rows"]) == 8
    assert physical["backend"] == "2.1.2.14"
    for row in physical["rows"]:
        decisions, execution_path, images = run.paths(row["root"], row["budget"])
        execution = json.loads(execution_path.read_text())
        assert row["execution_sha256"] == sha(execution_path)
        assert row["images_sha256"] == sha(images)
        assert row["decision_sha256"] == [sha(path) for path in decisions]
        actions = []
        for step, path in enumerate(decisions):
            decision = json.loads(path.read_text())
            assert decision["runner_sha256"] == sha(Path(run.__file__))
            assert decision["token_counts"]["prediction_token_samples"] == row["budget"]
            assert decision["chain"][:5] == execution["rows"][step]["actions"]
            actions.extend(decision["chain"][:5])
        assert actions == execution["executed_actions"]
