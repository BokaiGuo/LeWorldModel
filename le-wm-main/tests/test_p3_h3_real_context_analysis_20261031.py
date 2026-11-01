import json

import pytest

from experiments.spectral_reachability import p3_h3_real_context_analysis_20261031 as analysis


def test_saved_replay_deduplicates_and_preserves_censoring():
    result = analysis.summarize(json.loads(analysis.REPLAY.read_text()),
                                json.loads(analysis.GROUPS.read_text()))
    assert result["unique_complete_action_pairs"] == 13
    assert result["unique_complete_trajectories"] == 26
    assert result["censored_logical_pairs"] == 10
    assert result["refreshed_lower_mse"] == 26
    assert result["P3_admission"] is False


def test_duplicate_disagreement_fails_closed():
    replay = json.loads(analysis.REPLAY.read_text())
    groups = json.loads(analysis.GROUPS.read_text())
    replay["rows"][0]["actual_goal_cost"] += 1
    with pytest.raises(ValueError, match="duplicate replay differs"):
        analysis.summarize(replay, groups)
