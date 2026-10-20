import numpy as np

from experiments.gdd_wm.audit_train_only_teacher_capacity import (
    _bilinear_transfer_features,
    _feature_vector,
    _fit_train_pca,
    _goalframe_action_summary,
    _low_rank_bilinear_features,
    _macro_root_metrics,
    _pair_consistency,
    _proposal_relative_action_summaries,
    _root_metrics,
    _shuffle_predicted_delta_within_cells,
    _wilson_lower_bound,
)


def _row() -> dict:
    return {
        "source_state_raw": [0.0, 0.0, 0.0, 0.0],
        "goal_raw": [0.0, 1.0],
        "action_sequence": [[1.0, 0.0], [0.0, 1.0]],
        "source_latent_model": [0.2, -0.1],
        "goal_latent_model": [0.1, 0.4],
    }


def test_goalframe_action_summary_uses_source_goal_frame() -> None:
    summary = _goalframe_action_summary(_row())
    expected = np.asarray(
        [0.5, -0.5, 0.5, 0.5, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0],
        dtype=np.float64,
    )
    np.testing.assert_allclose(summary, expected)


def test_feature_vector_keeps_latent_delta_and_declared_action_frame() -> None:
    row = _row()
    predicted = np.asarray([0.4, 0.3], dtype=np.float64)
    raw = _feature_vector(row, predicted, "raw")
    goal_relative = _feature_vector(row, predicted, "goal_relative")
    assert raw.shape == goal_relative.shape == (16,)
    np.testing.assert_allclose(goal_relative[:2], row["source_latent_model"])
    np.testing.assert_allclose(goal_relative[2:4], predicted - np.asarray(row["source_latent_model"]))
    assert not np.allclose(raw[-12:], goal_relative[-12:])


def test_proposal_relative_action_chart_is_cell_local() -> None:
    first = np.stack([np.zeros(12), np.ones(12)], axis=0)
    second = 100.0 + 4.0 * first
    normalized = _proposal_relative_action_summaries(
        np.concatenate([first, second], axis=0),
        ["root_a", "root_a", "root_b", "root_b"],
    )
    np.testing.assert_allclose(normalized[:2], normalized[2:])
    np.testing.assert_allclose(normalized[:2].mean(axis=0), 0.0)
    np.testing.assert_allclose(normalized[:2].std(axis=0), 1.0)


def test_wilson_support_threshold_accounts_for_finite_candidate_count() -> None:
    empirical = 62 / 64
    lower = _wilson_lower_bound(62, 64, confidence=0.90)
    assert 0.85 < lower < empirical


def test_goal_latent_relative_conditioning_is_explicit_and_runtime_safe() -> None:
    row = _row()
    predicted = np.asarray([0.4, 0.3], dtype=np.float64)
    conditioned = _feature_vector(row, predicted, "goal_relative", "latent_relative")
    assert conditioned.shape == (20,)
    np.testing.assert_allclose(
        conditioned[16:18],
        np.asarray(row["goal_latent_model"]) - np.asarray(row["source_latent_model"]),
    )
    np.testing.assert_allclose(
        conditioned[18:20], predicted - np.asarray(row["goal_latent_model"]),
    )


def test_context_cells_are_macro_averaged_without_merging_candidate_sets() -> None:
    cells = {
        "context_a::root_1": {"pairwise_consistency": 0.25, "regret": 2.0},
        "context_b::root_1": {"pairwise_consistency": 0.75, "regret": 4.0},
        "context_a::root_2": {"pairwise_consistency": 0.50, "regret": 1.0},
    }
    mapping = {
        "context_a::root_1": "root_1",
        "context_b::root_1": "root_1",
        "context_a::root_2": "root_2",
    }
    result = _macro_root_metrics(cells, mapping)
    assert result["root_1"] == {"pairwise_consistency": 0.5, "regret": 3.0}
    assert result["root_2"] == {"pairwise_consistency": 0.5, "regret": 1.0}


def test_cost_readout_selects_lowest_predicted_cost() -> None:
    # The readout target is a cost.  A directionally correct diagnostic must
    # choose candidate ``a`` (predicted cost 1.0), not ``b`` (predicted cost
    # 9.0), even though the latter would be selected by argmax.
    metrics = _root_metrics(
        scores=np.asarray([1.0, 9.0]),
        costs=np.asarray([2.0, 8.0]),
        roots=["root", "root"],
        mask=np.asarray([True, True]),
        margin=1.0e-3,
    )
    assert metrics["root"]["regret"] == 0.0


def test_cost_pair_consistency_uses_lower_is_better_direction() -> None:
    assert _pair_consistency([(1.0, 2.0), (3.0, 4.0)], margin=1.0e-3) == 1.0
    assert _pair_consistency([(3.0, 2.0), (1.0, 4.0)], margin=1.0e-3) == 0.0


def test_prediction_shuffle_is_cell_local_and_preserves_source_action() -> None:
    features = np.asarray([
        [1.0, 2.0, 10.0, 20.0, 100.0],
        [1.0, 2.0, 30.0, 40.0, 200.0],
        [3.0, 4.0, 50.0, 60.0, 300.0],
        [3.0, 4.0, 70.0, 80.0, 400.0],
    ])
    shuffled = _shuffle_predicted_delta_within_cells(
        features, ["a", "a", "b", "b"], latent_dimension=2
    )
    np.testing.assert_allclose(shuffled[:, :2], features[:, :2])
    np.testing.assert_allclose(shuffled[:, 4:], features[:, 4:])
    np.testing.assert_allclose(shuffled[:2, 2:4], features[[1, 0], 2:4])
    np.testing.assert_allclose(shuffled[2:, 2:4], features[[3, 2], 2:4])


def test_bilinear_transfer_separates_no_dynamics_and_predicted_terms() -> None:
    source = np.asarray([[2.0, 3.0]])
    delta = np.asarray([[5.0, 7.0]])
    action = np.asarray([[11.0, 13.0]])
    no_dynamics = _bilinear_transfer_features(
        source, delta, action, include_predicted_dynamics=False
    )
    full = _bilinear_transfer_features(
        source, delta, action, include_predicted_dynamics=True
    )
    np.testing.assert_allclose(no_dynamics, [[11.0, 13.0, 22.0, 26.0, 33.0, 39.0]])
    assert full.shape == (1, 16)
    np.testing.assert_allclose(full[:, : no_dynamics.shape[1]], no_dynamics)


def test_train_pca_does_not_read_outcome_rows() -> None:
    train = np.asarray([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.5, 0.1, 0.0],
        [-0.5, -0.1, 0.0],
    ])
    values = np.concatenate([train, [[0.0, 0.0, 1.0e6]]], axis=0)
    mask = np.asarray([True, True, True, True, False])
    mean, components = _fit_train_pca(values, mask, rank=2)
    np.testing.assert_allclose(mean, np.mean(train, axis=0))
    assert components.shape == (2, 3)
    # The giant outcome-only third coordinate cannot rotate the train chart.
    np.testing.assert_allclose(components[:, 2], 0.0, atol=1.0e-12)


def test_low_rank_bilinear_has_bounded_width_and_expected_order() -> None:
    source = np.asarray([[2.0, 3.0]])
    delta = np.asarray([[5.0, 7.0]])
    action = np.asarray([[11.0, 13.0]])
    identity = np.eye(2)
    kwargs = {
        "source_mean": np.zeros(2),
        "source_components": identity,
        "delta_mean": np.zeros(2),
        "delta_components": identity,
    }
    no_dynamics = _low_rank_bilinear_features(
        source, delta, action, include_predicted_dynamics=False, **kwargs,
    )
    full = _low_rank_bilinear_features(
        source, delta, action, include_predicted_dynamics=True, **kwargs,
    )
    np.testing.assert_allclose(no_dynamics, [[11.0, 13.0, 22.0, 26.0, 33.0, 39.0]])
    assert full.shape == (1, 16)
    np.testing.assert_allclose(full[:, : no_dynamics.shape[1]], no_dynamics)


def test_low_rank_bilinear_goal_chart_is_train_runtime_symmetric() -> None:
    source = np.asarray([[2.0, 3.0]])
    delta = np.asarray([[5.0, 7.0]])
    goal_relative = np.asarray([[11.0, 13.0]])
    action = np.asarray([[17.0, 19.0]])
    identity = np.eye(2)
    kwargs = {
        "source_mean": np.zeros(2),
        "source_components": identity,
        "delta_mean": np.zeros(2),
        "delta_components": identity,
        "goal": goal_relative,
        "goal_mean": np.zeros(2),
        "goal_components": identity,
    }
    no_dynamics = _low_rank_bilinear_features(
        source, delta, action, include_predicted_dynamics=False, **kwargs,
    )
    full = _low_rank_bilinear_features(
        source, delta, action, include_predicted_dynamics=True, **kwargs,
    )
    # [action, source-action, goal, goal-action] followed by the predicted
    # delta blocks and goal-relative x predicted-delta interaction.
    assert no_dynamics.shape == (1, 12)
    assert full.shape == (1, 26)
    np.testing.assert_allclose(no_dynamics[0, :2], action[0])
    np.testing.assert_allclose(no_dynamics[0, 2:6], np.outer(source[0], action[0]).reshape(-1))
    np.testing.assert_allclose(no_dynamics[0, 6:8], goal_relative[0])
    np.testing.assert_allclose(no_dynamics[0, 8:12], np.outer(goal_relative[0], action[0]).reshape(-1))
