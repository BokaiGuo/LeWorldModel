import numpy as np
import pytest

from experiments.spectral_reachability.codex_p1_goal_pusher_metric_20261030 import translated_goal


def test_translation_preserves_other_pixels():
    image = np.full((224, 224, 3), 255, dtype=np.uint8)
    image[100:104, 100:104] = [65, 105, 225]
    image[99, 100] = [176, 185, 218]
    image[20, 20] = [1, 2, 3]
    moved = translated_goal(image, 8, 0)
    assert np.all(moved[100:104, 100:104] == 255)
    assert np.all(moved[100:104, 108:112] == [65, 105, 225])
    assert np.array_equal(moved[99, 108], [176, 185, 218])
    assert np.array_equal(moved[20, 20], image[20, 20])


def test_rejects_overlapping_sprite():
    image = np.full((224, 224, 3), 255, dtype=np.uint8)
    image[100:104, 100:104] = [65, 105, 225]
    image[101, 109] = [1, 2, 3]
    with pytest.raises(Exception):
        translated_goal(image, 8, 0)
