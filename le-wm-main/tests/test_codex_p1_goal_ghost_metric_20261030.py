import numpy as np
import pytest

from experiments.spectral_reachability.codex_p1_goal_ghost_metric_20261030 import translated_goal


def scene():
    rgb = np.full((224, 224, 3), 255, dtype=np.uint8)
    rgb[100:106, 100:106] = [65, 105, 225]
    return rgb


def test_marker_preserves_existing_scene_and_partial_overlap():
    rgb = scene()
    rgb[100:103, 109:112] = [80, 80, 80]
    before = rgb.copy()
    out = translated_goal(rgb, 8, 0)
    nonwhite = np.any(before != 255, axis=-1)
    assert np.array_equal(rgb, before)
    assert np.array_equal(out[nonwhite], before[nonwhite])
    assert np.count_nonzero(np.any(out != before, axis=-1)) >= 8
    assert np.array_equal(out[100:103, 109:112], before[100:103, 109:112])


def test_rejects_clipped_or_fully_occluded_marker():
    rgb = scene()
    with pytest.raises(ValueError):
        translated_goal(rgb, 130, 0)
    rgb[100:106, 108:114] = [80, 80, 80]
    with pytest.raises(ValueError):
        translated_goal(rgb, 8, 0)
