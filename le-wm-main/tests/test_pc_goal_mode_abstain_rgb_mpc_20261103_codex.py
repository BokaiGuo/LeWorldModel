import pytest

from experiments.pc_goal_mode_abstain_rgb_mpc_20261103_codex import context_ticks


def test_context_clock_tracks_real_feedback_frames():
    assert context_ticks(16) == (5, 10, 15)
    assert context_ticks(21) == (10, 15, 20)
    assert context_ticks(26) == (15, 20, 25)
    with pytest.raises(ValueError):
        context_ticks(20)
