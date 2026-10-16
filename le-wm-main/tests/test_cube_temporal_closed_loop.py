import numpy as np

from experiments.run_cube_temporal_closed_loop import (
    _prelude_action,
    _rank_consensus_score,
    _robust_rank_score,
    _rgb_block_goal_error,
    _support_exploration_action,
)


def test_closed_loop_prelude_is_fixed_and_does_not_read_state():
    assert np.allclose(_prelude_action(0), [0, 0, -1, 0, 0])
    assert np.allclose(_prelude_action(14), [0, 0, -1, 0, 0])
    assert np.allclose(_prelude_action(15), [0, 0, 0, 0, 1])
    assert np.allclose(_prelude_action(19), [0, 0, 0, 0, 1])
    assert np.allclose(_prelude_action(20), [0, 0, 0, 0, 0])


def test_support_exploration_is_fixed_four_direction_cycle():
    candidates = np.asarray([
        [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [-1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0], [0, -1, 0, 0, 0], [0, 0, -1, 0, 0],
        [0, 0, 0, 1, 0], [0, 0, 0, 0, -1],
    ], dtype=np.float32)
    np.testing.assert_allclose(_support_exploration_action(0, candidates), candidates[1])
    np.testing.assert_allclose(_support_exploration_action(4, candidates), candidates[1])


def test_rgb_feedback_error_uses_physical_red_component():
    current = np.zeros((32, 32, 3), dtype=np.uint8)
    goal = np.zeros((32, 32, 3), dtype=np.uint8)
    current[10:14, 8:12, 0] = 220
    goal[10:14, 20:24, 0] = 220
    error, valid = _rgb_block_goal_error(current, goal)
    assert valid
    assert error > 0.5


def test_rank_consensus_is_scale_invariant_and_finite():
    latent = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
    rgb = np.asarray([100.0, 200.0, 400.0, 800.0], dtype=np.float32)
    import torch

    score = _rank_consensus_score(torch.from_numpy(latent), torch.from_numpy(rgb))
    np.testing.assert_allclose(score.numpy(), 2.0 * _robust_rank_score(torch.from_numpy(latent)).numpy())
    assert np.isfinite(score.numpy()).all()
