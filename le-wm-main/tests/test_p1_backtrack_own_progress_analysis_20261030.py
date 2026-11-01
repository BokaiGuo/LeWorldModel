import math

import pytest

from experiments.spectral_reachability.p1_backtrack_own_progress_analysis_20261030 import pose_error


def test_position_and_circular_angle():
    assert pose_error([0, 0, 3, 4, -math.pi + .1, 0, 0],
                      [0, 0, 0, 0, math.pi - .1, 0, 0]) == pytest.approx((5, .2))


def test_rejects_bad_state():
    with pytest.raises(ValueError, match="finite"):
        pose_error([0] * 6, [0] * 7)
    with pytest.raises(ValueError, match="finite"):
        pose_error([float("nan")] * 7, [0] * 7)
