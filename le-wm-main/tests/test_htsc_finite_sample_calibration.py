import numpy as np

from experiments.decision_benchmark.audit_htsc_finite_sample_calibration import (
    _frozen_latent_action_basis_inputs,
    simulate_fixed_vs_adaptive_selection_shift,
    validate_fixed_bank_protocol,
)
from experiments.decision_benchmark.task_contrast import (
    calibrate_contrast_bound,
    calibrate_root_score_bound,
    certify_candidate_bank,
    certify_candidate_bank_regret_budget,
    evaluate_contrast_bound,
    finite_sample_marginal_coverage,
)


def test_htsc_finite_sample_rank_requires_nine_roots_for_ninety_percent_alpha():
    insufficient = finite_sample_marginal_coverage(8, alpha=0.1)
    assert insufficient["finite_sample_rank"] == 9
    assert insufficient["finite_bound_available"] is False
    assert insufficient["finite_sample_marginal_coverage_lower_bound"] is None
    sufficient = finite_sample_marginal_coverage(9, alpha=0.1)
    assert sufficient["finite_sample_rank"] == 9
    assert sufficient["finite_bound_available"] is True
    assert sufficient["finite_sample_marginal_coverage_lower_bound"] == 0.9


def test_htsc_insufficient_calibration_fails_closed_for_scalar_and_joint_contrast_bounds():
    scalar = calibrate_root_score_bound(np.arange(8, dtype=np.float64), alpha=0.1)
    assert scalar["status"] == "insufficient_calibration"
    assert scalar["bound"] is None
    assert scalar["finite_bound_available"] is False
    real = np.repeat(np.arange(8, dtype=np.float64), 2)
    predicted = np.zeros_like(real)
    groups = np.repeat(np.arange(8), 2)
    contrast = calibrate_contrast_bound(real, predicted, groups, np.arange(8), alpha=0.1)
    assert contrast["status"] == "insufficient_calibration"
    assert contrast["bound"] is None
    assert contrast["finite_sample_marginal_coverage_lower_bound"] is None


def test_htsc_rejects_an_incomplete_declared_calibration_panel():
    real = np.asarray([0.0, 1.0, 0.0])
    predicted = np.asarray([0.1, 0.9, 0.2])
    groups = np.asarray(["complete", "complete", "singleton"])
    report = calibrate_contrast_bound(
        real, predicted, groups, ["complete", "singleton", "missing"], alpha=.2,
    )
    assert report["status"] == "incomplete_calibration_panel"
    assert report["bound"] is None
    assert report["missing_calibration_groups"] == ["missing"]
    assert report["singleton_calibration_groups"] == ["singleton"]


def test_htsc_joint_candidate_bank_regret_bound_matches_covered_root():
    predicted = np.asarray([0.0, 0.4, 2.0])
    certificate = certify_candidate_bank(predicted, 0.5, support_passed=True)
    assert certificate["status"] == "probe"
    assert certificate["joint_regret_bound_status"] == "valid_on_simultaneous_fixed_bank_coverage_event"
    assert np.isclose(certificate["joint_regret_upper_bound"], 0.1)
    coverage = evaluate_contrast_bound(
        np.asarray([0.1, 0.0, 2.0]), predicted, np.asarray(["root"] * 3), 0.5,
    )
    root = coverage["roots"][0]
    assert root["covered"] is True
    assert np.isclose(root["realized_regret_of_predicted_winner"], 0.1)
    assert np.isclose(root["joint_regret_upper_bound_on_coverage_event"], 0.1)
    assert root["regret_bound_holds_when_covered"] is True
    assert coverage["covered_root_regret_bound_all_hold"] is True


def test_htsc_regret_budget_can_release_a_near_tied_fixed_bank_without_relabeling_it_unique():
    certificate = certify_candidate_bank_regret_budget(
        np.asarray([0.0, 0.4, 2.0]), 0.5,
        regret_budget=0.1, support_passed=True,
    )
    assert certificate["status"] == "permit"
    assert certificate["unique_winner_certificate_status"] == "probe"
    assert np.isclose(certificate["joint_regret_upper_bound"], 0.1)
    assert np.isclose(certificate["regret_budget"], 0.1)


def test_htsc_regret_budget_fails_closed_when_the_fixed_bank_bound_is_too_large():
    certificate = certify_candidate_bank_regret_budget(
        np.asarray([0.0, 0.1, 2.0]), 0.5,
        regret_budget=0.1, support_passed=True,
    )
    assert certificate["status"] == "probe"
    assert certificate["reason"] == "joint_regret_bound_exceeds_declared_budget"


def test_htsc_adaptive_candidate_generation_withholds_regret_bound():
    certificate = certify_candidate_bank(
        np.asarray([0.0, 2.0, 3.0]), 0.5, support_passed=True,
        candidate_bank_mode="adaptive",
    )
    assert certificate["status"] == "abstain"
    assert certificate["joint_regret_upper_bound"] is None
    assert certificate["joint_regret_bound_status"] == "unavailable_adaptive_candidate_generation"


def test_htsc_support_gate_blocks_release_without_deleting_fixed_bank_math():
    certificate = certify_candidate_bank(
        np.asarray([0.0, 2.0, 3.0]), 0.5, support_passed=False,
    )
    assert certificate["status"] == "abstain"
    assert certificate["joint_regret_upper_bound"] == 0.0
    assert certificate["joint_regret_bound_status"] == (
        "valid_on_simultaneous_fixed_bank_coverage_event_but_release_support_gate_failed"
    )


def test_htsc_independent_root_coverage_reports_sampling_interval():
    report = evaluate_contrast_bound(
        np.asarray([0.0, 1.0, 0.0, 1.0]),
        np.asarray([0.0, 1.1, 0.0, 2.0]),
        np.asarray(["covered", "covered", "missed", "missed"]),
        0.2,
    )
    interval = report["empirical_root_coverage_wilson_95_interval"]
    assert report["empirical_root_coverage"] == 0.5
    assert interval["lower"] < 0.5 < interval["upper"]


def test_htsc_fixed_vs_adaptive_selection_shift_stress_is_explicit_and_fail_closed():
    report = simulate_fixed_vs_adaptive_selection_shift(trials=800, seed=17)
    assert report["finite_bound_trial_count"] == 800
    assert report["fixed_declared_bank_empirical_coverage"] >= 0.86
    assert (
        report["adaptive_proposal_bank_naive_empirical_coverage"]
        < report["fixed_declared_bank_empirical_coverage"] - 0.10
    )
    assert report["adaptive_runtime_policy"] == "must abstain: adaptive_candidate_generation_not_certified"


def test_htsc_fixed_bank_validation_rejects_cross_root_action_mutation():
    def row(root: str, split: str, candidate: int, action: float) -> dict[str, object]:
        return {
            "instance_id": root,
            "split": split,
            "candidate_id": candidate,
            "action_sequence": [[action, 0.0], [action, 0.0]],
        }

    development = [
        row("train", "train", 0, 0.0), row("train", "train", 1, 1.0),
        row("cal", "calibration", 0, 0.0), row("cal", "calibration", 1, 1.0),
    ]
    external = [
        row("fresh", "external_outcome", 0, 0.0), row("fresh", "external_outcome", 1, 1.0),
    ]
    report = validate_fixed_bank_protocol(development, external)
    assert report["root_overlap_count"] == 0
    external[-1]["action_sequence"] = [[2.0, 0.0], [2.0, 0.0]]
    try:
        validate_fixed_bank_protocol(development, external)
    except ValueError as error:
        assert "common action sequence" in str(error)
    else:
        raise AssertionError("fixed-bank validator accepted a cross-root action mutation")


def test_htsc_external_input_builder_never_requires_realized_successor_fields():
    rows = [{
        "source_latent_model": [0.0] * 32,
        "action_sequence": [[0.1, -0.1]] * 20,
    }]
    values = _frozen_latent_action_basis_inputs(rows)
    assert values.shape == (1, 48)
