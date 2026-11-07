import numpy as np
import pytest

from experiments.pc_tworoom_goal_render_alignment_20261107_codex import difference


def test_difference_preserves_uint8_signed_delta():
    a = np.asarray([0, 255], dtype=np.uint8)
    b = np.asarray([255, 255], dtype=np.uint8)
    result = difference(a, b)
    assert result == {"mae": 127.5, "max": 255., "unequal_fraction": 0.5}


def test_difference_rejects_broadcasting_or_nonfinite():
    with pytest.raises(ValueError):
        difference([1, 2], [1])
    with pytest.raises(ValueError):
        difference([float("nan")], [1])
