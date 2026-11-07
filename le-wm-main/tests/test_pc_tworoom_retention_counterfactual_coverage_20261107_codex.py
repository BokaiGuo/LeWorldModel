import pytest

from experiments.pc_tworoom_retention_counterfactual_coverage_20261107_codex import analyze_root


def test_counterfactual_labels_keep_model_selection_separate_from_oracle():
    plan = {"row": 3, "selected": {"lookahead": [0, 1], "descriptor": [0, 2]},
            "outcomes": {"lookahead": {"cost": 1.0}, "descriptor": {"cost": 2.0}}}
    original = {"outcomes": {"lookahead": {"ever_success": False}, "descriptor": {"ever_success": False}}}
    representatives = [{"prefix": i, "ever_success": i == 2, "model_cost": 1.5} for i in range(8)]
    result = analyze_root(plan, original, representatives)
    assert result["oracle_any"] is True
    assert result["lookahead_success"] is False
    assert result["additional_success_cases"] == [
        {"prefix": 2, "lookahead_kept": False, "descriptor_kept": True,
         "lower_model_cost_than_lookahead": False, "lower_model_cost_than_descriptor": True}
    ]


def test_counterfactual_requires_exact_prefix_coverage():
    with pytest.raises(ValueError):
        analyze_root({}, {}, [{"prefix": 0}])
