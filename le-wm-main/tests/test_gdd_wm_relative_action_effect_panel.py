from experiments.gdd_wm.audit_relative_action_effect_panel import _bootstrap_effects
from experiments.gdd_wm.aggregate_relative_action_effect_panel import aggregate
from experiments.gdd_wm.aggregate_feature_family_panel import aggregate as aggregate_feature_families
import pytest
import json


def test_relative_effect_bootstrap_is_paired_and_deterministic():
    def metrics(regrets, consistencies):
        return {
            "root_metrics": {
                "outcome": {
                    root: {
                        "regret": regret,
                        "execution_consistency": consistency,
                        "comparable_pair_count": 3,
                    }
                    for root, regret, consistency in zip(
                        ("r0", "r1"), regrets, consistencies
                    )
                }
            }
        }

    report = {
        "variants": {
            "root_centered_source_delta_action": metrics((0.4, 0.2), (0.5, 0.6)),
            "candidate": metrics((0.2, 0.1), (0.7, 0.8)),
        }
    }
    first = _bootstrap_effects(report, samples=100, seed=7)
    second = _bootstrap_effects(report, samples=100, seed=7)
    effect = first["effects"]["outcome"]["candidate"]
    assert first == second
    assert effect["consistency_gain"]["point"] == 0.2
    assert effect["regret_reduction"]["point"] == pytest.approx(0.15)
    assert effect["consistency_gain"]["root_count"] == 2


def test_aggregate_keeps_case_and_root_uncertainty_separate(tmp_path):
    def variant(offset):
        return {
            "root_metrics": {
                "outcome": {
                    "r0": {"regret": 0.4 - offset, "execution_consistency": 0.5 + offset},
                    "r1": {"regret": 0.2 - offset, "execution_consistency": 0.6 + offset},
                }
            }
        }

    panel = {
        "panel_member_order": ["case_a", "case_b"],
        "panel": {
            "case_a": {"variants": {"base": variant(0.0), "goal": variant(0.1)}},
            "case_b": {"variants": {"base": variant(0.0), "goal": variant(0.2)}},
        },
    }
    panel_path = tmp_path / "panel.json"
    output_path = tmp_path / "aggregate.json"
    panel_path.write_text(json.dumps(panel), encoding="utf-8")
    report = aggregate(
        panel_path,
        output_path,
        baseline="base",
        variant="goal",
        samples=200,
        seed=11,
    )
    assert report["protocol"]["case_unit"] == "architecture_by_seed_checkpoint"
    assert report["aggregate"]["consistency_gain"]["all_case_points_positive"] is True
    assert report["aggregate"]["regret_reduction"]["all_case_points_positive"] is True
    assert report["aggregate"]["consistency_gain"]["two_level_case_and_root_bootstrap"]["point"] == pytest.approx(0.15)
    assert output_path.exists()


def test_feature_family_aggregate_reports_paired_outcome_effect(tmp_path):
    def family(regret, consistency):
        return {
            "root_metrics": {
                "outcome": {
                    "r0": {"regret": regret, "execution_consistency": consistency},
                    "r1": {"regret": regret, "execution_consistency": consistency},
                }
            }
        }

    panel = {
        "panel_member_order": ["case_a", "case_b"],
        "panel": {
            name: {
                "feature_families": {
                    "source_plus_delta": family(0.5, 0.5),
                    "source_delta_action": family(0.3 if name == "case_a" else 0.4, 0.7),
                }
            }
            for name in ("case_a", "case_b")
        },
    }
    panel_path = tmp_path / "feature_panel.json"
    output_path = tmp_path / "feature_aggregate.json"
    panel_path.write_text(json.dumps(panel), encoding="utf-8")
    report = aggregate_feature_families(
        panel_path,
        output_path,
        baseline="source_plus_delta",
        samples=200,
        seed=13,
    )
    result = report["families"]["source_delta_action"]["regret_reduction"]
    assert result["all_case_points_positive"] is True
    assert result["two_level_case_and_root_bootstrap"]["point"] == pytest.approx(0.15)
