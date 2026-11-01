from copy import deepcopy

import pytest

from experiments.spectral_reachability import codex_p1_root1_candidate_consumer_20261101_exclusive as c


def test_candidate_consumer_rejects_altered_actions_and_identity():
    point = {"state": [1], "snapshot": {"a": 1}, "pixel_sha256": "x",
             "goal": [2], "goal_rgb_sha256": "y"}
    original = [[[0.0, 0.0]] * 5 for _ in range(3)]
    opt = [[[0.1, 0.0]] * 5 for _ in range(3)]
    entry = c.make_expected(original, opt, point)
    c.check_entry(entry, original, opt, point)
    bad = deepcopy(entry)
    bad["hold"][0][0][0] = 0.2
    with pytest.raises(ValueError):
        c.check_entry(bad, original, opt, point)
    bad = deepcopy(entry)
    bad["required_cp5"]["snapshot"]["a"] = 2
    with pytest.raises(ValueError):
        c.check_entry(bad, original, opt, point)


def test_original_plan_selection_checks_role_and_stage_ties():
    p0 = [[[0.0, 0.0]] * 5 for _ in range(3)]
    p1 = [[[0.25, 0.0]] * 5 for _ in range(3)]
    event = {"role": "greedy", "bank": [p0, p1], "stage_scores": [1.0, 1.0],
             "original_selected_index": 1, "full_selected": p0}
    c.check_selection(event, "greedy")
    event["full_selected"] = p1
    with pytest.raises(ValueError):
        c.check_selection(event, "greedy")
