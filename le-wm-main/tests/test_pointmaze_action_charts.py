import numpy as np

from experiments.empirical_spectral_audit.pointmaze_action_charts import (
    action_blocks_for_chart,
    goal_relative_action_blocks,
)


def test_goal_relative_chart_rotates_world_x_to_source_goal_axis() -> None:
    rows = [{"source_state_raw": [0.0, 0.0, 0.0, 0.0]}]
    actions = np.zeros((1, 4, 10), dtype=np.float32)
    actions[0, :, 0::2] = 1.0
    transformed = goal_relative_action_blocks(rows, actions, goal=np.asarray([0.0, 1.0]))
    np.testing.assert_allclose(transformed[0, :, 0::2], 0.0, atol=1e-6)
    np.testing.assert_allclose(transformed[0, :, 1::2], -1.0, atol=1e-6)


def test_goal_relative_chart_preserves_shape_and_uses_fixed_goal_fallback() -> None:
    rows = [
        {"source_state_raw": [0.4, 0.4, 0.0, 0.0]},
        {"source_state_raw": [0.6, 0.6, 0.0, 0.0]},
    ]
    actions = np.ones((2, 40), dtype=np.float32)
    transformed = goal_relative_action_blocks(rows, actions)
    assert transformed.shape == actions.shape
    assert np.all(np.isfinite(transformed))


def test_hybrid_chart_keeps_raw_carrier_and_adds_goal_relative_features() -> None:
    rows = [{"source_state_raw": [0.0, 0.0, 0.0, 0.0]}]
    actions = np.zeros((1, 4, 10), dtype=np.float32)
    actions[0, :, 0::2] = 1.0
    hybrid = action_blocks_for_chart(rows, actions, "hybrid")
    assert hybrid.shape == (1, 4, 20)
    np.testing.assert_allclose(hybrid[0, :, :10], actions[0])
