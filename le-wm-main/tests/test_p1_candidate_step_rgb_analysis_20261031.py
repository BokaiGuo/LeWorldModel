from copy import deepcopy
import json

import pytest

from experiments.spectral_reachability import p1_candidate_step_rgb_analysis_20261031 as analysis


@pytest.fixture(scope="module")
def data():
    assert analysis.source.base.sha(analysis.INPUT) == analysis.SHA
    return json.loads(analysis.INPUT.read_bytes())


def test_fixed_roster_and_same_time_success(data):
    result = analysis.analyze(data)
    assert len(result["comparisons"]) == 5
    assert all(not item["winner_stopped_before"]
               and item["success_real_goal_mse"] < item["winner_real_same_time_goal_mse"]
               and item["winner_model_terminal_score"] < item["success_model_terminal_score"]
               for item in result["comparisons"])
    assert result["P1_admission"] is False


def test_missing_or_duplicate_candidate_and_false_success_rejected(data):
    bad = deepcopy(data)
    bad["rows"][1] = deepcopy(bad["rows"][0])
    with pytest.raises(ValueError, match="candidate identity"):
        analysis.analyze(bad)
    bad = deepcopy(data)
    candidate = next(row for row in bad["rows"] if row["kind"] != 0 and row["success"])
    candidate["steps"][-1]["terminated"] = False
    with pytest.raises(ValueError, match="success termination"):
        analysis.analyze(bad)
