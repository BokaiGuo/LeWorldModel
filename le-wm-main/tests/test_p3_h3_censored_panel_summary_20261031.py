import copy
import json

import pytest

from experiments.spectral_reachability import p3_h3_real_context_physical_summary_20261031 as summary


def test_fixed_panel_deduplicates_and_keeps_guard_censoring():
    path = summary.ROOT / "results/p3_h3_censored_panel_summary_20261031_RUN01.json"
    data = json.loads(path.read_bytes())
    rows = data["rows"]
    assert len(rows) == 52
    assert sum(r["new_endpoint"] is None for r in rows) == 4
    assert all(r["new_minus_old"] is None for r in rows if r["new_endpoint"] is None)
    assert data["unique_action_pairs"] == 26
    assert data["complete_unique_pairs"] == 24
    assert sum(data["direction_counts_new_minus_old"]["object_position_error"].values()) == 24


def test_aggregate_refuses_silent_duplicate_disagreement():
    path = summary.ROOT / "results/p3_h3_censored_panel_summary_20261031_RUN01.json"
    data = json.loads(path.read_bytes())
    rows = copy.deepcopy(data["rows"])
    rows += [{**copy.deepcopy(rows[0]), "identity": {**rows[0]["identity"], "padding": True}},
             {**copy.deepcopy(rows[1]), "identity": {**rows[1]["identity"], "padding": True}}]
    groups = {}
    for row in rows:
        if not row["identity"]["padding"]:
            groups.setdefault(row["pair_key"], []).append(row)
    duplicate = next(group for group in groups.values() if len(group) > 1)
    duplicate[1]["new_status"] = "fabricated"
    with pytest.raises(ValueError, match="duplicate outcome mismatch"):
        summary.aggregate(rows)
