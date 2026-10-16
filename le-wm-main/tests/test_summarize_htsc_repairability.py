import json

from experiments.decision_benchmark.summarize_htsc_repairability import summarize


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summary_preserves_mechanism_positive_and_abstain(tmp_path):
    coverage = {
        "calibration": {"bound": 1.0},
        "conformal_outcome_coverage": {
            "empirical_root_coverage": 0.9,
            "covered_root_count": 9,
            "outcome_root_count": 10,
        },
        "selective_outcome": {"released_root_count": 0},
    }
    _write(tmp_path / "htsc_fixed_bank_alpha010.json", coverage)
    _write(tmp_path / "shared_world_htsc_frozen_alpha010.json", coverage)
    _write(tmp_path / "intervention_effect_frozen_latent_action_certified.json", {
        "ensemble_task_contrast": {
            "outcome_selection": {"mean_regret": 0.2},
            "bank_only_control": {"mean_regret": 0.1},
            "paired_regret_gain_over_controls": {
                "bank_only": {"estimate": -0.1, "lower": -0.2, "upper": -0.01},
            },
        },
    })
    _write(tmp_path / "shared_world_htsc_rgb_candidate_adapter_alpha010.json", {
        "rgb_contrast_outcome": {"contrast_error_mean": 1.0},
        "physical_contrast_outcome": {"inversion_rate": 0.5},
        "selective_outcome": {"all_root_mean_regret": 0.2, "released_root_count": 0},
    })
    positive = {"estimate": 0.1, "lower": 0.01, "upper": 0.2}
    _write(tmp_path / "shared_world_intervention_effect_certified.json", {
        "ensemble_task_contrast": {
            "outcome_selection": {"mean_regret": 0.03},
            "bank_only_control": {"mean_regret": 0.08},
            "no_dynamics_control": {"mean_regret": 0.2},
            "shuffled_prediction_control": {"mean_regret": 0.16},
            "paired_regret_gain_over_controls": {
                "bank_only": positive, "no_dynamics": positive,
                "shuffled_prediction": positive,
            },
            "new_candidate_control": {"paired_regret_gain_over_shuffled": positive},
            "matched_order_pair_audit": {
                "paired_regret_gain_over_bank_only": positive,
                "outcome_left_better_fraction": {"0-1": 0.5},
                "bounded_regret_certificate": {},
            },
            "certificate_preview": {"operational_release_count": 0},
            "bounded_regret_certificate": {},
        },
    })
    report = summarize(tmp_path, tmp_path / "summary.json")
    assert report["shared_world_repair"]["mechanism_positive"] is True
    assert report["permission"]["status"] == "abstain"
    assert report["protocol"]["checkpoint_split_provenance"] == {
        "train": 54, "calibration": 14, "outcome": 32,
    }
