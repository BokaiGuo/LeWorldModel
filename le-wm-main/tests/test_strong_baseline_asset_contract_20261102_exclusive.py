import pytest

from experiments.spectral_reachability import (
    strong_baseline_asset_contract_20261102_exclusive as m,
)


def test_catalog_has_four_reported_only_assets_and_no_cross_paper_ranking():
    catalog = m.build_catalog()
    assert set(catalog["assets"]) == {"GRASP", "Hi-LeWM", "LeWM", "PlanWAM"}
    assert catalog["cross_paper_ranking_allowed"] is False
    assert catalog["local_reproduction"] is False
    assert catalog["neural_calls"] == catalog["physics_calls"] == 0
    assert {v["evidence_kind"] for v in catalog["assets"].values()} == {"paper_reported"}


def test_reported_counts_and_domains_are_preserved():
    assets = m.build_catalog()["assets"]
    assert {name: row["reported_cells"] for name, row in assets.items()} == {
        "GRASP": 112,
        "Hi-LeWM": 106,
        "LeWM": 8,
        "PlanWAM": 12,
    }
    assert assets["GRASP"]["outcome_domain"] == "learned_world_model_only"
    assert assets["PlanWAM"]["outcome_domain"] == "autonomous_driving"
    assert assets["GRASP"]["metric_unit"] == "success_percent_and_time_seconds"
    assert assets["PlanWAM"]["metric_unit"] == "domain_specific_scores"


@pytest.mark.parametrize(
    "name",
    ["GRASP", "Hi-LeWM", "LeWM", "PlanWAM"],
)
def test_each_asset_requires_local_protocol_alignment_before_comparison(name):
    asset = m.build_catalog()["assets"][name]
    assert asset["local_comparison_status"] == "blocked_protocol_mismatch_or_unknown"
    assert asset["required_local_bindings"]
    assert asset["paper_value_usage"] == "context_only"


def test_rejects_cross_paper_ranking_and_outcome_fields():
    invalid_catalog = m.build_catalog()
    invalid_catalog["cross_paper_ranking_allowed"] = True
    with pytest.raises(ValueError, match="ranking"):
        m.validate_catalog(invalid_catalog)
    with pytest.raises(ValueError, match="outcome"):
        m.validate_author_value({"method": "GRASP", "success": 1.0})


def test_unknown_is_not_comparable_and_reported_values_are_not_local_values():
    asset = m.build_catalog()["assets"]["LeWM"]
    assert m.comparison_status(asset, {"checkpoint": "unknown"}) == "unknown"
    assert m.comparison_status(asset, {
        "checkpoint": "x",
        "task": "PushT",
        "metric": "success_rate",
        "feedback": "open_loop",
        "budget": "unknown",
    }) == "blocked"
