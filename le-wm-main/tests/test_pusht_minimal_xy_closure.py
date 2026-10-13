import numpy as np

from experiments.run_pusht_minimal_xy_closure import xy


def test_xy_selects_agent_and_block_positions():
    state_features = np.arange(8.0)
    np.testing.assert_array_equal(xy(state_features), np.arange(4.0))
