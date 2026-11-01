import copy

import pytest

from experiments.spectral_reachability import p1_adam_budget_metric_disagreement_20261031 as analysis


def test_actual_panel_counts_and_stopped_endpoints():
    result = analysis.analyze(analysis.source.pairing.read(analysis.INPUT, analysis.INPUT_SHA))
    assert len(result["pairs"]) == 72
    for counter in result["counts"].values():
        assert sum(counter.values()) == 24
    assert any(row["metrics"]["object_position_error"] is None for row in result["pairs"])


def test_rejects_bad_roster_and_nonfinite():
    panel = analysis.source.pairing.read(analysis.INPUT, analysis.INPUT_SHA)
    bad = copy.deepcopy(panel)
    bad["rows"][0]["index"] = 3
    with pytest.raises(ValueError, match="row roster"):
        analysis.analyze(bad)
    bad = copy.deepcopy(panel)
    bad["rows"][0]["budgets"]["10"]["model_best_score"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        analysis.analyze(bad)
