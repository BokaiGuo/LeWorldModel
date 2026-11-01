import json
from pathlib import Path


def test_saved_pair_real_24_and_budget_axes():
    root = Path(__file__).resolve().parents[1]
    value = json.loads((root/"results/codex_p1_cp5_twenty_physical_saved_20261030_v1_CODEX01.json").read_bytes())
    assert value["status"] == "24_pairs_saved_verified" and len(value["rows"]) == 24
    assert value["compute_budgets_differ"] and not value["equal_budget_certified"]
    for i, row in enumerate(value["rows"]):
        assert row["index"] == i
        for arm, predict, backward in (("two_step", 6, 2), ("twenty_step", 42, 20)):
            entry = row[arm]
            assert entry["status"] == "probe_completed" and entry["accepted_suffix_steps"] == 10
            counts = entry["planning_resources"]["planning"]["charged_upper_bounds"]
            assert counts["predict_calls"] == predict and counts["backward_calls"] == backward
            assert entry["physical_resources"]["public_step"]["completed"] == 25
