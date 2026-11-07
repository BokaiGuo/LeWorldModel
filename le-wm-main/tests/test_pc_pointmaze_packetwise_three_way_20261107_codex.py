import numpy as np
import pytest

from experiments.pc_pointmaze_packetwise_three_way_20261107_codex import strict_mse


def test_strict_mse_rejects_broadcasting_and_nonfinite():
    with pytest.raises(ValueError):
        strict_mse(np.zeros((2, 1)), np.zeros((2,)))
    with pytest.raises(ValueError):
        strict_mse([np.nan], [0])
    assert strict_mse([0, 2], [0, 0]) == 2
