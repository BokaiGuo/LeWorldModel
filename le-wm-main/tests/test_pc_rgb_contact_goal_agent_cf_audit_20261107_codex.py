import json

import numpy as np

from experiments.pc_rgb_contact_goal_agent_cf_audit_20261107_codex import count_improved, summarize


def test_summary_counts_identity_changes_not_regret_changes():
    rows = [{"results": {
        "original": {"predicted": {"selected": 0, "physical_regret": .4}},
        "alternative": {"predicted": {"selected": 1, "physical_regret": .4}},
    }}]
    result = summarize(rows, "predicted")
    assert result["selection_changed"] == 1
    assert result["mean_regret_original"] == result["mean_regret_alternative"]


def test_numpy_root_changes_are_json_serializable():
    count = count_improved([np.float64(-0.2), np.float64(0.0), np.float64(0.1)])
    assert count == 1
    assert json.loads(json.dumps({"count": count})) == {"count": 1}
