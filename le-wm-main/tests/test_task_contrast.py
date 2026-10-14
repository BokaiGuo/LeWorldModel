import numpy as np

from experiments.decision_benchmark.task_contrast import (
    calibrate_contrast_bound,
    certify_pair,
    pairwise_task_contrast_audit,
)


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
    assert calibration["bound"] >= 0.0
    assert certify_pair(0.0, 1.0, calibration["bound"], support_passed=False)["status"] == "abstain"
    assert certify_pair(0.0, 1.0, None, support_passed=True)["status"] == "probe"


def test_certificate_permits_only_when_predicted_margin_exceeds_bound():
    report = certify_pair(0.0, 2.0, 1.0, support_passed=True, candidate_ids=("good", "bad"))
    assert report["status"] == "permit"
    assert report["preferred_candidate"] == "good"
