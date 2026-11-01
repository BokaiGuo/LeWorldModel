import copy

import pytest

from experiments.spectral_reachability import p1_split_budget_crossover_online_20261031 as subject


def test_duplicate_panel_is_rejected():
    panel = {"anchors": [{"index": i} for i in range(24)],
             "banks": [[{"index": i}] for i in range(8)]}
    with pytest.raises(ValueError, match="duplicate completed crossover panel"):
        subject.reject_duplicate_panel(copy.deepcopy(panel), panel)


def test_distinct_panel_is_not_called_identical():
    panel = {"anchors": [{"index": i} for i in range(24)],
             "banks": [[{"index": i}] for i in range(8)]}
    changed = copy.deepcopy(panel)
    changed["banks"][0][0]["index"] = 99
    subject.reject_duplicate_panel(changed, panel)


def test_missing_existing_roster_fails_closed():
    with pytest.raises(ValueError, match="existing panel roster"):
        subject.reject_duplicate_panel({"anchors": [], "banks": []}, {})
