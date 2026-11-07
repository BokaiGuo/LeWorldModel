import numpy as np
import pytest

from experiments.pc_camera_multiview_dev_20261106_codex import choose


def test_ensemble_uses_mean_and_disagreement_abstains():
    scores = [[3, 1, 2], [3, 1, 2], [3, 1, 2], [3, 1, 2], [3, 5, 0]]
    assert choose(scores) == {"centered": 1, "multi": 2, "guard": 0, "zero": 0}


def test_unanimous_and_bad_inputs():
    assert choose(np.tile([2, 3, 1], (5, 1)))["guard"] == 2
    with pytest.raises(ValueError):
        choose([[np.nan, 1, 2]] * 5)
    with pytest.raises(ValueError):
        choose([[1, 2, 3]])
