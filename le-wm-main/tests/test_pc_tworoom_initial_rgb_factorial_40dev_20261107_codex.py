import numpy as np
import pytest

from experiments.pc_tworoom_initial_rgb_factorial_40dev_20261107_codex import pixel_stats, select


def test_pixel_stats_tracks_channel_difference():
    first = np.zeros((2, 3, 3), dtype=np.uint8)
    second = first.copy()
    second[1, 1, :2] = 1
    out = pixel_stats(first, second)
    assert out["changed_pixels"] == 1 and out["max"] == 1
    assert out["mean"] == pytest.approx(2/18)


def test_choice_tie_and_invalid():
    scores = np.ones(32)
    scores[3] = 0
    scores[4] = 0
    assert select(scores) == (3, 0.)
    scores[0] = np.nan
    with pytest.raises(ValueError):
        select(scores)
