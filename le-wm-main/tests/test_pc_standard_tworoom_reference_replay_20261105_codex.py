import numpy as np

from experiments.pc_standard_tworoom_reference_replay_20261105_codex import select_rows


def test_select_rows_matches_native_eval():
    episodes = np.repeat(np.arange(20), 30)
    steps = np.tile(np.arange(30), 20)
    valid = np.flatnonzero(steps <= 4)
    expected = np.sort(valid[np.random.default_rng(42).choice(len(valid) - 1, size=50, replace=False)])
    np.testing.assert_array_equal(select_rows({"ep_idx": episodes, "step_idx": steps}), expected)
