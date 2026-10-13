import numpy as np

from experiments.empirical_spectral_audit.train_pointmaze_learned_adapter_panel import (
    HORIZONS,
    _features,
    _planning_risk,
    _planning_risk_with_calibration_gate,
    _task_direction,
    _task_readout_with_center,
)


def _row(horizon, source, target):
    return {
        "source_state_raw": [float(source), 0.0],
        "latent_by_horizon": {
            str(horizon): {"target_state_raw": [float(target), 0.0]}
            for horizon in HORIZONS
        },
    }


def test_planning_risk_uses_each_horizon_readout_without_name_error():
    rows = [_row(1, 0.0, 1.0), _row(1, 0.0, 0.0)]
    predictions = {h: np.asarray([[float(h)], [0.0]], dtype=np.float64) for h in HORIZONS}
    decoders = {h: np.ones((1, 2), dtype=np.float64) for h in HORIZONS}
    centers = {h: np.zeros(1, dtype=np.float64) for h in HORIZONS}

    risk, details = _planning_risk(
        predictions,
        {},
        rows,
        decoders,
        np.asarray([1.0, 0.0]),
        centers,
    )

    assert np.isfinite(risk)
    assert details["candidate_horizons"] == list(HORIZONS)
    assert sum(details["selected_horizon_counts"].values()) == len(rows)


def test_task_readout_returns_train_only_center_and_does_not_center_outcome_batch():
    train_targets = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    train_states = np.zeros((3, 2), dtype=np.float64)
    target_states = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    decoder, center = _task_readout_with_center(train_targets, train_states, target_states)

    assert np.allclose(center, [2.0, 0.0])
    heldout = np.asarray([[10.0, 0.0], [11.0, 0.0]])
    # The fitted center is invariant to which outcome batch is evaluated.
    assert np.allclose((heldout - center) @ decoder, [[8.0, 0.0], [9.0, 0.0]], atol=5e-2)


def test_task_direction_uses_within_transition_targets_not_episode_row_order():
    source = np.asarray([[0.0, 0.0], [10.0, 0.0], [0.0, 0.0]])
    targets = source + np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

    direction = _task_direction(source, targets)

    assert direction[0] > 0.99
    assert abs(direction[1]) < 1e-8


def test_feature_builder_uses_shared_raw_actions_not_export_specific_normalization():
    rows = []
    for index in range(2):
        rows.append({
            "source_latent": [0.0] * 8,
            "action_blocks_raw": [[0.1] * 10] * 4,
            "action_blocks_normalized": [[1.7] * 10] * 4,
            "source_state_raw": [0.0, 0.0],
            "latent_by_horizon": {
                str(h): {"target": [0.0] * 8, "target_state_raw": [0.0, 0.0]}
                for h in HORIZONS
            },
        })
    mean = np.zeros(8)
    basis = np.eye(8)

    x, _, _, _, _ = _features(rows, mean, basis)

    assert np.allclose(x[0, 8:], 0.1)


def test_calibration_gate_falls_back_to_h1_when_long_horizon_error_is_large():
    calibration_rows = [_row(1, 0.0, 0.0), _row(1, 0.0, 0.0)]
    outcome_rows = [_row(1, 0.0, 1.0), _row(1, 0.0, 0.0)]
    predictions = {h: np.asarray([[0.0], [0.0]], dtype=np.float64) for h in HORIZONS}
    calibration_predictions = {h: np.asarray([[0.0], [0.0]], dtype=np.float64) for h in HORIZONS}
    # Make h=2/h=4 appear attractive in the outcome prediction, but make the
    # calibration target disagree.  The gate must keep h=1 available and
    # select it rather than trusting the long-horizon score.
    predictions[2] = np.asarray([[10.0], [10.0]])
    predictions[4] = np.asarray([[10.0], [10.0]])
    calibration_predictions[2] = np.asarray([[10.0], [10.0]])
    calibration_predictions[4] = np.asarray([[10.0], [10.0]])
    decoders = {h: np.ones((1, 2), dtype=np.float64) for h in HORIZONS}
    centers = {h: np.zeros(1, dtype=np.float64) for h in HORIZONS}

    risk, details = _planning_risk_with_calibration_gate(
        predictions,
        calibration_predictions,
        outcome_rows,
        calibration_rows,
        decoders,
        np.asarray([1.0, 0.0]),
        centers,
    )

    assert np.isfinite(risk)
    assert details["selected_horizon_counts"]["1"] == len(outcome_rows)


def test_transfer_permission_rejects_high_action_inverse_risk_fail_closed():
    calibration_rows = [_row(1, 0.0, 0.0) for _ in range(4)]
    outcome_rows = [_row(1, 0.0, 1.0), _row(1, 0.0, 0.0)]
    predictions = {h: np.zeros((2, 1), dtype=np.float64) for h in HORIZONS}
    calibration_predictions = {h: np.zeros((4, 1), dtype=np.float64) for h in HORIZONS}
    decoders = {h: np.ones((1, 2), dtype=np.float64) for h in HORIZONS}
    centers = {h: np.zeros(1, dtype=np.float64) for h in HORIZONS}

    _, details = _planning_risk_with_calibration_gate(
        predictions,
        calibration_predictions,
        outcome_rows,
        calibration_rows,
        decoders,
        np.asarray([1.0, 0.0]),
        centers,
        calibration_action_risk=0.20,
        train_task_scale=1.0,
    )

    assert details["planning_permission"] == "rejected_fail_closed"
    assert "action_effect_not_identifiable" in details["permission_reasons"]
    assert details["candidate_horizons_allowed"] == [1]
    assert details["selected_horizon_counts"]["1"] == len(outcome_rows)


def test_transfer_permission_accepts_calibrated_effect_and_limits_bad_horizon():
    calibration_rows = [_row(1, 0.0, 0.0) for _ in range(4)]
    outcome_rows = [_row(1, 0.0, 1.0), _row(1, 0.0, 0.0)]
    predictions = {h: np.zeros((2, 1), dtype=np.float64) for h in HORIZONS}
    calibration_predictions = {h: np.zeros((4, 1), dtype=np.float64) for h in HORIZONS}
    decoders = {h: np.ones((1, 2), dtype=np.float64) for h in HORIZONS}
    centers = {h: np.zeros(1, dtype=np.float64) for h in HORIZONS}

    _, details = _planning_risk_with_calibration_gate(
        predictions,
        calibration_predictions,
        outcome_rows,
        calibration_rows,
        decoders,
        np.asarray([1.0, 0.0]),
        centers,
        calibration_action_risk=0.01,
        train_task_scale=1.0,
    )

    assert details["planning_permission"] == "accepted"
    assert details["candidate_horizons_allowed"] == [1, 2, 4]

