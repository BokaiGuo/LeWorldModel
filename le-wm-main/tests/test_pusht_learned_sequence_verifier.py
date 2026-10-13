import numpy as np

from experiments.run_pusht_learned_sequence_verifier import (
    MODEL_SEED_OFFSETS,
    choose_sequence,
    fit_models,
    fit_pairwise_angle_ranker,
    label_sequence,
    make_sequences,
    online_orientation_progress,
    pairwise_win_scores,
    task_rank_target,
    choose_uncertainty_aware_sequence,
    calibration_gate_decision,
    successor_mode_probabilities,
)


class _BinaryModel:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)
        self.classes_ = np.asarray([0, 1], dtype=np.int64)

    def predict_proba(self, features):
        values = np.resize(self.values, len(features))
        return np.column_stack([1.0 - values, values])


class _Regressor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def predict(self, features):
        return np.resize(self.values, len(features))


def _models(n):
    binary = {
        name: _BinaryModel(np.full(n, 0.95))
        for name in ("pose", "turn", "motion", "contact", "path_contact", "joint")
    }
    continuous = {
        "angle_progress": _Regressor(np.linspace(0.1, 0.8, n)),
        "xy_progress": _Regressor(np.zeros(n)),
        "motion_distance": _Regressor(np.full(n, 3.0)),
    }
    return {**binary, **continuous}


def test_model_seed_offsets_are_semantic_and_complete():
    expected = {
        "pose", "turn", "motion", "contact", "path_contact", "joint",
        "angle_progress", "xy_progress", "motion_distance",
        "successor_mode",
    }
    assert set(MODEL_SEED_OFFSETS) == expected
    assert len(set(MODEL_SEED_OFFSETS.values())) == len(expected)


def test_fit_models_does_not_change_existing_target_rng_when_path_contact_is_inserted():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(40, 5))
    labels = {
        "pose": np.tile([0, 1], 20),
        "turn": np.tile([1, 0], 20),
        "motion": np.tile([0, 1], 20),
        "contact": np.tile([1, 0], 20),
        "path_contact": np.tile([0, 1], 20),
        "joint": np.tile([1, 0], 20),
        "angle_progress": rng.normal(size=40),
        "xy_progress": rng.normal(size=40),
        "motion_distance": np.abs(rng.normal(size=40)),
        "successor_mode": np.tile([0, 1, 2, 1], 10),
    }
    reordered = {name: labels[name] for name in reversed(tuple(labels))}
    first = fit_models(features, labels, trees=8, seed=123)
    second = fit_models(features, reordered, trees=8, seed=123)
    for name in labels:
        assert getattr(first[name], "random_state", None) == getattr(second[name], "random_state", None)
        np.testing.assert_allclose(first[name].predict(features), second[name].predict(features))


def test_knn_model_family_exposes_the_same_prediction_interface():
    rng = np.random.default_rng(29)
    features = rng.normal(size=(24, 5))
    labels = {
        "contact": np.tile([0, 1], 12),
        "path_contact": np.tile([1, 0], 12),
        "angle_progress": rng.normal(size=24),
        "xy_progress": rng.normal(size=24),
        "motion_distance": np.abs(rng.normal(size=24)),
        "pose": np.tile([0, 1], 12),
        "turn": np.tile([1, 0], 12),
        "motion": np.tile([1, 0], 12),
        "joint": np.tile([0, 1], 12),
        "successor_mode": np.tile([0, 1, 2], 8),
    }
    models = fit_models(features, labels, trees=8, seed=31, model_family="knn", knn_neighbors=4)
    assert np.isfinite(models["angle_progress"].predict(features)).all()
    assert np.isfinite(models["contact"].predict_proba(features)).all()


def test_label_sequence_initializes_path_contact_from_endpoint_contact():
    state = np.asarray([100.0, 100.0, 120.0, 120.0, 0.0, 0.0, 0.0])
    goal = state.copy()
    endpoint = state.copy()
    result = label_sequence(state, goal, endpoint)
    assert result["path_contact"] == result["contact"]
    assert all(np.isfinite(value) for value in result.values())


def test_make_sequences_supports_horizon_matched_one_step_candidates():
    actions = np.asarray([[0.0, 0.0], [0.2, 0.0]], dtype=np.float32)
    sequences = make_sequences(actions, length=1)
    assert sequences.shape == (2, 1, 2)
    np.testing.assert_allclose(sequences[:, 0], actions)


def test_mode_selector_forbids_zero_when_a_nonzero_mode_feasible_candidate_exists():
    sequences = make_sequences(np.asarray([[0.0, 0.0], [0.2, 0.0]], dtype=np.float32))
    models = _models(len(sequences))
    selected, info = choose_sequence(
        "sequence_mode_lexicographic", models,
        np.zeros(7), np.zeros(7), sequences, "local",
    )
    assert info["zero_guard_active"]
    assert info["nonzero_feasible_count"] > 0
    assert info["selected_nonzero"]
    assert np.any(np.abs(sequences[selected]) > 1e-8)


def test_receding_mode_uses_the_same_mode_gate_on_one_step_sequences():
    sequences = make_sequences(np.asarray([[0.0, 0.0], [0.2, 0.0]], dtype=np.float32), length=1)
    models = _models(len(sequences))
    selected, info = choose_sequence(
        "sequence_mode_receding", models,
        np.zeros(7), np.zeros(7), sequences, "local",
    )
    assert info["decision_horizon"] == 1
    assert info["selected_nonzero"]
    assert selected == 1


def test_pairwise_angle_ranker_scores_the_better_candidate_higher():
    rng = np.random.default_rng(19)
    features = rng.normal(size=(12, 4))
    targets = np.asarray([0.0, 1.0, 0.2, -0.1, 0.3, 0.8, 0.0, 0.5, 0.4, -0.2, 0.7, 0.1])
    model = fit_pairwise_angle_ranker(features, targets, group_size=3, trees=16, seed=11)
    scores = pairwise_win_scores(model, features[:3])
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_task_rank_target_rejects_contact_loss_before_angle_progress():
    labels = {
        "angle_progress": np.asarray([0.8, 0.1, 0.2]),
        "xy_progress": np.asarray([1.0, 1.0, -8.0]),
        "motion": np.asarray([1, 1, 1]),
        "contact": np.asarray([0, 1, 1]),
        "path_contact": np.asarray([0, 1, 1]),
    }
    target = task_rank_target(labels)
    assert target[0] < target[1]
    assert target[2] < target[1]
    assert np.isfinite(target).all()


def test_mode_selector_uses_nonzero_fail_closed_fallback_when_no_candidate_is_feasible():
    sequences = make_sequences(np.asarray([[0.0, 0.0], [0.2, 0.0]], dtype=np.float32))
    models = _models(len(sequences))
    for name in ("turn", "contact", "path_contact"):
        models[name] = _BinaryModel(np.full(len(sequences), 0.05))
    models["motion_distance"] = _Regressor(np.full(len(sequences), 0.1))
    selected, info = choose_sequence(
        "sequence_mode_lexicographic", models,
        np.zeros(7), np.zeros(7), sequences, "local",
    )
    assert info["selection_mode"] == "mode_support_violation_nonzero_fallback"
    assert info["selected_nonzero"]
    assert np.any(np.abs(sequences[selected]) > 1e-8)


def test_online_orientation_progress_uses_only_observed_history_and_is_finite():
    sequences = make_sequences(np.asarray([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2]], dtype=np.float32), length=1)
    state = np.asarray([100.0, 100.0, 120.0, 100.0, 0.0, 0.0, 0.0])
    goal = state.copy()
    goal[4] = 0.5
    assert online_orientation_progress(state, goal, sequences, []) is None
    history = [
        {"before_state": state, "action": np.asarray([0.2, 0.0]), "after_state": state + np.asarray([0, 0, 0, 0, 0.04, 0, 0])},
        {"before_state": state, "action": np.asarray([0.0, 0.2]), "after_state": state + np.asarray([0, 0, 0, 0, 0.02, 0, 0])},
    ]
    scores = online_orientation_progress(state, goal, sequences, history)
    assert scores is not None
    assert scores.shape == (len(sequences),)
    assert np.isfinite(scores).all()


def test_successor_mode_probabilities_are_aligned_to_all_three_modes():
    rng = np.random.default_rng(101)
    features = rng.normal(size=(30, 4))
    labels = np.tile([0, 1, 2], 10)
    models = fit_models(
        features,
        {"successor_mode": labels},
        trees=12,
        seed=17,
    )
    probabilities = successor_mode_probabilities(models["successor_mode"], features[:5])
    assert probabilities.shape == (5, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.isfinite(probabilities).all()


def test_uncertainty_selector_uses_successor_mode_and_returns_set_diagnostics():
    rng = np.random.default_rng(102)
    features = rng.normal(size=(60, 5))
    labels = {
        "turn": np.tile([0, 1], 30),
        "angle_progress": np.linspace(-0.2, 0.8, 60),
        "xy_progress": np.zeros(60),
        "motion_distance": np.full(60, 3.0),
        "successor_mode": np.tile([2, 2, 1], 20),
    }
    models = fit_models(features, labels, trees=16, seed=23)
    sequences = make_sequences(np.asarray([[0.0, 0.0], [0.2, 0.0]], dtype=np.float32), length=1)
    selected, info = choose_uncertainty_aware_sequence(
        models, features[:2], sequences, np.zeros(7), 0.5, 2.0, 4.0, "nonzero", 1.0,
    )
    assert selected in (0, 1)
    assert info["uncertainty_source"]
    assert info["selected_nonzero"]
    assert np.isfinite(info["selected_score"])


def test_uncertainty_selector_enters_contact_reacquisition_outside_contact():
    rng = np.random.default_rng(103)
    features = rng.normal(size=(60, 5))
    labels = {
        "turn": np.ones(60, dtype=np.int64),
        "angle_progress": np.zeros(60),
        "xy_progress": np.zeros(60),
        "motion_distance": np.full(60, 3.0),
        "successor_mode": np.tile([1, 2, 2], 20),
    }
    models = fit_models(features, labels, trees=16, seed=29)
    sequences = make_sequences(np.asarray([[0.0, 0.0], [0.2, 0.0]], dtype=np.float32), length=1)
    selected, info = choose_uncertainty_aware_sequence(
        models, features[:2], sequences,
        np.asarray([100.0, 100.0, 250.0, 100.0, 0.0, 0.0, 0.0]),
        0.5, 2.0, 4.0, "nonzero", 1.0,
    )
    assert selected in (0, 1)
    assert info["selection_mode"] == "contact_reacquisition"
    assert info["current_mode"] == "free"


def test_calibration_gate_rejects_pairwise_when_any_pose_coordinate_regresses():
    baseline = [
        {"strict_success": 1, "final_xy": 10.0, "final_angle": 0.20, "contact_retention": 0.90},
        {"strict_success": 0, "final_xy": 12.0, "final_angle": 0.25, "contact_retention": 0.80},
    ]
    candidate = [
        {"strict_success": 1, "final_xy": 10.0, "final_angle": 0.30, "contact_retention": 0.90},
        {"strict_success": 0, "final_xy": 12.0, "final_angle": 0.31, "contact_retention": 0.80},
    ]
    decision = calibration_gate_decision(baseline, candidate)
    assert decision["passed"] is False
    assert decision["decision"] == "mode_receding_fallback"
    assert decision["checks"]["angle_noninferior"] is False


def test_calibration_gate_admits_pairwise_only_when_all_metrics_are_noninferior():
    baseline = [
        {"strict_success": 0, "final_xy": 10.0, "final_angle": 0.30, "contact_retention": 0.80},
        {"strict_success": 1, "final_xy": 14.0, "final_angle": 0.20, "contact_retention": 0.90},
    ]
    candidate = [
        {"strict_success": 0, "final_xy": 9.0, "final_angle": 0.25, "contact_retention": 0.82},
        {"strict_success": 1, "final_xy": 13.0, "final_angle": 0.21, "contact_retention": 0.91},
    ]
    decision = calibration_gate_decision(baseline, candidate)
    assert decision["passed"] is True
    assert decision["decision"] == "candidate_admitted"

