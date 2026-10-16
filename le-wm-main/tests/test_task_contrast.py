import numpy as np

from experiments.decision_benchmark.task_contrast import (
    _split_conformal_upper_quantile,
    calibrate_contrast_bound,
    calibrate_root_score_bound,
    certify_candidate_bank,
    certify_pair,
    evaluate_contrast_bound,
    pairwise_task_contrast_audit,
)


def test_finite_sample_quantile_enumeration_for_declared_panel_sizes():
    expected_ranks = {8: 9, 10: 10, 20: 19, 40: 37}
    for calibration_size, expected_rank in expected_ranks.items():
        pooled = np.arange(calibration_size + 1, dtype=np.float64)
        covered = 0
        finite = True
        observed_ranks = set()
        for held_out in range(calibration_size + 1):
            calibration = np.delete(pooled, held_out)
            bound, rank = _split_conformal_upper_quantile(calibration, alpha=0.1)
            observed_ranks.add(rank)
            if bound is None:
                finite = False
            else:
                covered += int(pooled[held_out] <= bound)
        assert observed_ranks == {expected_rank}
        if calibration_size == 8:
            assert finite is False
            assert covered == 0
        else:
            assert finite is True
            assert covered / (calibration_size + 1) >= 0.9


def test_common_prediction_bias_cancels_in_task_contrast():
    real = np.asarray([0.0, 2.0, 1.0, 3.0])
    predicted = real + 10.0
    report = pairwise_task_contrast_audit(real, predicted, np.asarray(["r0", "r0", "r1", "r1"]))
    assert report["contrast_error_max"] == 0.0
    assert report["inversion_rate"] == 0.0


def test_contrast_audit_detects_task_direction_inversion():
    report = pairwise_task_contrast_audit(
        np.asarray([0.0, 2.0]), np.asarray([2.0, 0.0]), np.asarray(["root"] * 2),
        minimum_decision_margin=0.1,
    )
    assert report["inversion_count"] == 1
    assert report["contrast_error_max"] == 4.0


def test_calibration_is_root_level_and_certificate_fails_closed():
    calibration = calibrate_contrast_bound(
        np.asarray([0.0, 2.0, 1.0, 3.0]),
        np.asarray([0.1, 1.9, 1.2, 2.8]),
        np.asarray(["cal0", "cal0", "cal1", "cal1"]),
        ["cal0", "cal1"], alpha=0.1,
    )
    assert calibration["calibration_root_count"] == 2
    assert calibration["status"] == "insufficient_calibration"
    assert calibration["bound"] is None
    assert certify_pair(0.0, 1.0, calibration["bound"], support_passed=False)["status"] == "abstain"
    assert certify_pair(0.0, 1.0, None, support_passed=True)["status"] == "probe"


def test_finite_sample_rank_is_used_when_calibration_is_large_enough():
    roots = np.arange(10, dtype=np.float64)
    calibration = calibrate_contrast_bound(
        np.repeat([0.0, 1.0], 10),
        np.repeat([0.0, 1.0], 10),
        np.repeat(roots, 2),
        roots,
        alpha=0.1,
    )
    assert calibration["status"] == "ok"
    assert calibration["finite_sample_rank"] == 10
    assert calibration["bound"] == 0.0


def test_scalar_root_risk_uses_same_finite_sample_rule():
    calibration = calibrate_root_score_bound(np.arange(10, dtype=np.float64), alpha=0.1)
    assert calibration["status"] == "ok"
    assert calibration["finite_sample_rank"] == 10
    assert calibration["bound"] == 9.0


def test_certificate_permits_only_when_predicted_margin_exceeds_bound():
    report = certify_pair(0.0, 2.0, 1.0, support_passed=True, candidate_ids=("good", "bad"))
    assert report["status"] == "permit"
    assert report["preferred_candidate"] == "good"


def test_candidate_bank_certificate_checks_every_competitor():
    report = certify_candidate_bank(
        np.asarray([0.0, 2.0, 0.4]), 0.5,
        support_passed=True, candidate_ids=("best", "far", "close"),
    )
    assert report["status"] == "probe"
    assert report["reason"] == "at_least_one_competitor_not_separated_by_bound"
    assert report["preferred_candidate"] == "best"


def test_candidate_bank_certificate_fails_closed_without_finite_calibration():
    report = certify_candidate_bank(
        np.asarray([0.0, 2.0, 3.0]), None, support_passed=True,
    )
    assert report["status"] == "probe"
    assert report["reason"] == "contrast_bound_missing_or_insufficient_calibration"


def test_candidate_bank_certificate_rejects_adaptive_cem_claim():
    report = certify_candidate_bank(
        np.asarray([0.0, 2.0, 3.0]), 0.5, support_passed=True,
        candidate_bank_mode="adaptive",
    )
    assert report["status"] == "abstain"
    assert report["reason"] == "adaptive_candidate_generation_not_certified"


def test_independent_outcome_coverage_uses_root_maximum_error():
    report = evaluate_contrast_bound(
        np.asarray([0.0, 1.0, 0.0, 1.0]),
        np.asarray([0.0, 1.1, 0.0, 2.0]),
        np.asarray(["covered", "covered", "missed", "missed"]),
        0.2,
    )
    assert report["outcome_root_count"] == 2
    assert report["covered_root_count"] == 1
    assert report["empirical_root_coverage"] == 0.5


def test_outcome_coverage_is_undefined_without_finite_calibration():
    report = evaluate_contrast_bound(
        np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]), np.asarray(["r0", "r0"]), None,
    )
    assert report["status"] == "not_evaluable_insufficient_calibration"
    assert report["empirical_root_coverage"] is None
