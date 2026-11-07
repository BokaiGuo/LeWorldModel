import numpy as np
import pytest

from experiments.pc_pointmaze_archive_context_multistep_20261107_codex import mse


def test_mse_requires_matching_finite_features():
    assert mse([1, 2], [2, 4]) == 2.5
    with pytest.raises(ValueError):
        mse([1], [1, 2])
    with pytest.raises(ValueError):
        mse([np.nan], [1])
