import numpy as np

from experiments import run_pusht_probe_reobserve_select as evaluator


def test_fixed_bank_is_ordered_global_and_fingerprinted():
    bank = evaluator.fixed_action_bank(0.5)
    np.testing.assert_allclose(bank, [[0.0, 0.5], [0.0, -0.5]])
    assert evaluator.action_bank_sha256(bank) == evaluator.action_bank_sha256(bank.copy())
    assert evaluator.action_bank_sha256(bank) != evaluator.action_bank_sha256(bank[::-1])


def test_rgb_chart_uses_action_conditioned_observed_effects_only():
    chart = evaluator.fit_rgb_effect_chart(
        contexts=np.asarray([[0.0, 0.0], [0.1, 0.0], [2.0, 2.0], [2.1, 2.0]]),
        action_indices=np.asarray([0, 0, 1, 1]),
        effects=np.asarray([[2.0, 0.0], [1.8, 0.0], [0.1, 0.0], [0.2, 0.0]]),
    )
    prediction = evaluator.predict_rgb_effects(chart, np.asarray([0.05, 0.0]), neighbors=2)

    assert prediction["selected_index"] == 0
    assert prediction["predicted_effect_energy"][0] > prediction["predicted_effect_energy"][1]
    assert prediction["runtime_privileged_state_access"] is False
    assert prediction["runtime_successor_access"] is False


def test_gate_and_abstain_execute_the_identical_fallback_identity():
    gate = evaluator.calibrate_gate(
        [{"support_distance": 0.1, "predicted_contrast": 0.3} for _ in range(5)], alpha=0.2, minimum_roots=5
    )
    abstain = evaluator.select_or_abstain(
        {"selected_index": 1, "support_distance": 1.0, "predicted_contrast": 9.0}, gate, fallback_index=0
    )
    released = evaluator.select_or_abstain(
        {"selected_index": 1, "support_distance": 0.1, "predicted_contrast": 0.3}, gate, fallback_index=0
    )

    assert gate["status"] == "calibrated"
    assert abstain == {"selected_index": 0, "release": False, "reason": "insufficient_support"}
    assert released == {"selected_index": 1, "release": True, "reason": "calibrated_rgb_effect_selection"}


def test_insufficient_finite_calibration_fails_closed_without_selecting_local_action():
    gate = evaluator.calibrate_gate(
        [{"support_distance": 0.1, "predicted_contrast": 0.3} for _ in range(2)], alpha=0.1, minimum_roots=3
    )
    decision = evaluator.select_or_abstain(
        {"selected_index": 1, "support_distance": 0.0, "predicted_contrast": 99.0}, gate, fallback_index=0
    )

    assert gate["status"] == "insufficient_calibration"
    assert decision["selected_index"] == 0
    assert decision["release"] is False
    assert decision["reason"] == "insufficient_calibration"


def test_root_partition_overlap_is_rejected():
    root = evaluator.FrozenRoot(7, np.zeros(7), np.zeros(7))
    try:
        evaluator.validate_root_disjoint([root], [root], [])
    except RuntimeError as error:
        assert "root-disjoint" in str(error)
    else:
        raise AssertionError("overlapping root IDs must fail closed")


def test_paired_summary_reports_all_roots_not_released_subset():
    rows = [
        {"fallback": {"final_xy": 3.0, "strict_pose_success": False}, "local": {"final_xy": 2.0, "strict_pose_success": False}, "local_runtime": {"released": True}},
        {"fallback": {"final_xy": 1.0, "strict_pose_success": True}, "local": {"final_xy": 1.0, "strict_pose_success": True}, "local_runtime": {"released": False}},
    ]
    summary = evaluator.paired_summary(rows)

    assert summary["test_roots"] == 2
    assert summary["local_release_count"] == 1
    assert summary["local_wins"] == 1
    assert summary["ties"] == 1
    assert summary["mean_final_xy_delta_local_minus_fallback"] == -0.5
