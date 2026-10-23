import numpy as np
import pytest

from experiments.decision_benchmark.probe_conditioned_candidate_bank import (
    SCHEMA,
    action_hull_support_mask,
    build_probe_conditioned_candidate_bank,
    build_prior_anchored_residual_probe_conditioned_candidate_bank,
)


def _cardinal_probes() -> np.ndarray:
    return np.asarray([
        [0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, -0.5],
    ], dtype=np.float64)


def test_bank_uses_only_supported_goal_progressing_actions_and_has_audit_trace() -> None:
    probes = _cardinal_probes()
    # A local observed chart with a small lateral component, not a simulator
    # oracle.  Positive x is the RGB-derived current goal direction.
    effects = probes @ np.asarray([[0.9, 0.1], [0.0, 0.7]], dtype=np.float64)
    bank = build_probe_conditioned_candidate_bank(probes, effects, np.asarray([1.0, 0.0]))
    assert bank is not None
    assert len(bank.actions) >= 4
    assert np.all(action_hull_support_mask(probes, bank.actions))
    assert np.all(bank.predicted_effects[:, 0] > 0.0)
    trace = bank.trace_record()
    assert trace["schema"] == SCHEMA
    assert trace["privileged_state_or_outcome_used"] is False
    assert trace["construction_inputs"] == (
        "RGB_observed_probe_actions_RGB_observed_effects_RGB_goal_direction_only"
    )
    assert len(bank.trace_digest()) == 64


def test_bank_fails_closed_when_the_observed_chart_has_no_goal_progress() -> None:
    probes = _cardinal_probes()
    # Every observed action affects only the goal-orthogonal coordinate.
    # Reversing an action would be a valid repair in the previous fixture, so
    # it is not evidence that construction should abstain.
    effects = np.stack([np.zeros(len(probes)), probes[:, 1]], axis=1)
    assert build_probe_conditioned_candidate_bank(
        probes, effects, np.asarray([1.0, 0.0]),
    ) is None


def test_bank_fails_closed_after_a_probe_reaches_the_current_waypoint() -> None:
    probes = _cardinal_probes()
    assert build_probe_conditioned_candidate_bank(
        probes, probes, np.zeros(2, dtype=np.float64),
    ) is None


def test_bank_fails_closed_when_probe_actions_do_not_identify_an_affine_chart() -> None:
    probes = np.asarray([[0.5, 0.0], [0.25, 0.0], [-0.5, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="at least four"):
        build_probe_conditioned_candidate_bank(
            probes, probes, np.asarray([1.0, 0.0]), candidate_count=3,
        )
    assert build_probe_conditioned_candidate_bank(
        probes, probes, np.asarray([1.0, 0.0]), candidate_count=4,
    ) is None


def test_prior_anchored_residual_bank_removes_drift_and_keeps_zero_anchor_support() -> None:
    probes = _cardinal_probes()
    # Simulate an inertial baseline that would otherwise appear as a changing
    # affine intercept.  The measured residual remains action-identifiable.
    residuals = probes @ np.asarray([[0.30, 0.0], [0.0, 0.10]], dtype=np.float64)

    def baseline(actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float64)
        return np.repeat(np.asarray([[0.20, 0.0]], dtype=np.float64), len(values), axis=0)

    bank = build_prior_anchored_residual_probe_conditioned_candidate_bank(
        probes, residuals, np.asarray([1.0, 0.0]), baseline_effects=baseline,
    )
    assert bank is not None
    support = np.concatenate([np.zeros((1, 2), dtype=np.float64), probes], axis=0)
    assert np.all(action_hull_support_mask(support, bank.actions))
    assert np.all(bank.predicted_effects[:, 0] > 0.0)
    assert bank.diagnostics["zero_action_prior_anchor"] is True
    assert bank.trace_record()["construction_inputs"] == (
        "RGB_observed_probe_actions_RGB_observed_effect_residuals_"
        "RGB_goal_direction_calibration_dynamics_only"
    )
