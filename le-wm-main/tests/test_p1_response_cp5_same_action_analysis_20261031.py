import copy
import json

import pytest

from experiments.spectral_reachability import p1_response_cp5_same_action_analysis_20261031 as analysis


@pytest.fixture
def payload():
    assert analysis.source.sha(analysis.INPUT) == analysis.INPUT_SHA
    return json.loads(analysis.INPUT.read_bytes())


def test_complete_pair_denominators(payload):
    result = analysis.analyze(payload)
    assert result["completed"] == 13
    assert len(result["excluded"]) == 3
    assert len(result["pairs"]) == 5
    assert result["P1_admission"] is False


def test_modified_score_rejected(payload):
    data = copy.deepcopy(payload)
    data["rows"][0]["actual_goal_mse"] += 0.1
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        analysis.analyze(data)


def test_missing_censored_row_rejected(payload):
    data = copy.deepcopy(payload)
    data["excluded"].pop()
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        analysis.analyze(data)
