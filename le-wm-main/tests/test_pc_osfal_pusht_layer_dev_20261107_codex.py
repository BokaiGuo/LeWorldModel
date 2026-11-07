import numpy as np

from experiments.pc_osfal_pusht_layer_dev_20261107_codex import layer_metrics


def test_background_side_and_tie_aware_regret():
    f = np.zeros((2, 2, 5, 1))
    f[1] += 3
    f[:, 1] += 2
    f[:, :, 2] += 1
    f[:, :, 3] += 2
    physical = np.tile([2., 1., 0.], (2, 2, 1))
    row = layer_metrics(f, physical)
    assert row["background_mse"] == 9
    assert row["side_initial_mse"] == 4
    assert row["action_final_mse"] == 2
    assert row["conditions"][0]["regret"] == 2
