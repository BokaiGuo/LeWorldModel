import copy
import json

import pytest

from experiments.spectral_reachability import p3_h3_refresh_physical_attribution_20261031 as analysis


@pytest.fixture(scope="module")
def sources():
    return [json.loads(path.read_text()) for path in analysis.SOURCES]


def test_fixed_real_pairs_and_physical_ties(sources):
    result = analysis.analyze(*sources)
    assert result["complete_action_pairs"] == 13
    assert result["counts"]["refreshed_goal_cost"]["object_position_error"] == {
        "higher/lower": 1, "higher/tie": 3, "lower/lower": 5, "lower/tie": 4,
    }
    assert not result["P3_admission"]


def test_mismatched_task_and_open_score_rejected(sources):
    replay, physical = copy.deepcopy(sources)
    pair = next(g["representative"]["logical_pair"] for g in physical["groups"]
                if g["representative"]["physical_new_minus_old"] is not None)
    row = next(r for r in replay["rows"] if r["logical_pair"] == pair and r["variant"] == "new")
    row["root_slot"] += 1
    with pytest.raises(ValueError, match="task identity"):
        analysis.analyze(replay, physical)
    row["root_slot"] -= 1
    row["open_goal_cost"] += 1
    with pytest.raises(ValueError, match="open model score binding"):
        analysis.analyze(replay, physical)
