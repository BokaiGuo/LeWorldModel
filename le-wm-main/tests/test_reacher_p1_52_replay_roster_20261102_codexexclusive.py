import copy

import pytest

from experiments.spectral_reachability import reacher_p1_52_replay_roster_20261102_codexexclusive as roster


def rows():
    return [{"group": g, "seed": s, "arm": f"prefix_{i:02d}",
             "root_rgb_sha256": str(s), "goal_rgb_sha256": str(s),
             "cp5_rgb_sha256": str(i), "history": [i], "actions": [[i]] * 15}
            for g in ("L2", "Q") for s in (20263160, 20263161) for i in range(13)]


def test_full_roster_and_no_selection():
    sample = rows()
    roster.validate_roster(sample)
    with pytest.raises(Exception):
        roster.validate_roster(sample[:-1])
    with pytest.raises(Exception):
        roster.validate_roster(list(reversed(sample)))


@pytest.mark.parametrize("field", ["root_rgb_sha256", "goal_rgb_sha256", "cp5_rgb_sha256", "history", "actions"])
def test_reject_paired_context_or_prefix_drift(field):
    sample = copy.deepcopy(rows())
    sample[26][field] = "tampered"
    with pytest.raises(Exception):
        roster.validate_roster(sample)


def test_digest_is_order_invariant_and_content_sensitive():
    assert roster.digest({"a": 1, "b": 2}) == roster.digest({"b": 2, "a": 1})
    assert roster.digest({"a": 1}) != roster.digest({"a": 2})


def test_ranking_ties_and_censored_retained():
    panel = [{"prefix_index": i, "status": "complete", "predicted_goal_distance": i // 2,
              "actual_rgb_goal_distance": i, "physical_distance": i, "prediction_error": i}
             for i in range(13)]
    panel[2]["status"] = "crash"
    ranked = roster.rank_complete_panel(panel)
    assert len(ranked) == 13
    assert ranked[0]["predicted_goal_distance_rank"] == ranked[1]["predicted_goal_distance_rank"] == 1.5
    assert ranked[2]["physical_distance_rank"] is None
    assert ranked[3]["physical_distance_rank"] == 3
    panel[4]["physical_distance"] = float("nan")
    with pytest.raises(Exception):
        roster.rank_complete_panel(panel)
