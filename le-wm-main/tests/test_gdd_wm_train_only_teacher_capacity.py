import numpy as np

from experiments.gdd_wm.audit_train_only_teacher_capacity import (
    _feature_vector,
    _goalframe_action_summary,
    _macro_root_metrics,
    _root_metrics,
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
