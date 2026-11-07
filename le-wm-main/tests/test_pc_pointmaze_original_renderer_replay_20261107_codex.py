import numpy as np
import pytest

from experiments.pc_pointmaze_original_renderer_replay_20261107_codex import pixel_diff


def test_pixel_diff_no_unsigned_underflow():
    result = pixel_diff(np.array([0, 255], dtype=np.uint8), np.array([255, 254], dtype=np.uint8))
    assert result == {"mean": 128., "max": 255, "nonzero": 2, "elements": 2}


def test_pixel_diff_fails_closed_on_mismatch():
    with pytest.raises(ValueError):
        pixel_diff(np.array([0]), np.array([0]))


def test_separate_indexing_preserves_root_before_action():
    images = np.zeros((8, 3, 2, 6, 4, 4, 3), dtype=np.uint8)
    selected = np.take(images[::2], [2, 1], axis=1)[:, :, :, -1]
    assert selected.shape == (4, 2, 2, 4, 4, 3)
