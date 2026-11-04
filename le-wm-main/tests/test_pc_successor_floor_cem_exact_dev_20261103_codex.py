import json

import numpy as np
import pytest

from experiments import pc_successor_floor_cem_exact_dev_20261103_codex as experiment


def test_sealed_budget_selection_and_control_bindings():
    if not experiment.PLANS.exists() or not experiment.AUDIT.exists():
        pytest.skip("development artifacts unavailable")
    data = json.loads(experiment.PLANS.read_text())
    audit = json.loads(experiment.AUDIT.read_text())
    assert data["dependencies"] == experiment.dependencies()
    assert audit["plans_sha256"] == experiment.prior.sha(experiment.PLANS)
    assert len(data["rows"]) == len(audit["rows"]) == 120
    for row in data["rows"]:
        result = row["result"]
        assert row["tokens"] == sum(batch["tokens"] for batch in row["trace"])
        assert row["tokens"] <= 480
        assert result["scale_floor"] == .05
        before, selected = result["precomparison"], result["selected"]
        if row["arm"] != "full_cem":
            assert result["abstained"] == (result["comparison_score"] <= before["terminal_score"])
            assert selected["terminal_score"] == min(before["terminal_score"], result["comparison_score"])
            assert before["terminal_score"] <= min(x["terminal_score"] for x in result["programs"])
            comparison = np.asarray(result["comparison_chain"])
            np.testing.assert_array_equal(comparison[:5], np.zeros((5, 2)))
            np.testing.assert_array_equal(comparison[5:], np.asarray(before["chain"])[5:])
        else:
            assert selected == before
