import numpy as np
import pytest

from experiments.pc_tworoom_history_directional_20261107_codex import (
    history_window, pair_counts,
)


def test_history_window_uses_prior_frames_and_only_past_actions():
    episodes = np.full(16, 3)
    steps = np.arange(16)
    pixels = np.arange(16)
    actions = np.repeat(np.arange(16)[:, None], 2, axis=1) / 20
    frames, history = history_window(episodes, steps, pixels, actions, 10)
    np.testing.assert_array_equal(frames, [0, 5, 10])
    np.testing.assert_array_equal(history, actions[:10])


def test_history_window_rejects_cross_episode_clock_and_illegal_actions():
    episodes, steps = np.ones(11), np.arange(11)
    actions = np.zeros((11, 2))
    episodes[0] = 2
    with pytest.raises(ValueError, match="episode"):
        history_window(episodes, steps, steps, actions, 10)
    episodes[0] = 1
    steps[3] = 10
    with pytest.raises(ValueError, match="clock"):
        history_window(episodes, steps, steps, actions, 10)
    steps[3] = 3
    actions[0, 0] = 2
    with pytest.raises(ValueError, match="action"):
        history_window(episodes, steps, steps, actions, 10)


def test_pair_counts_reports_ties_and_correct_sign_separately():
    assert pair_counts([1., -1., 0., 1.], [2., 2., -1., 0.]) == {
        "agree": 1, "disagree": 1, "tie": 2}
