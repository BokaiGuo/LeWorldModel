import numpy as np
import pytest

from experiments.pc_pusht_goal_render_offset_probe_20261107_codex import image_error


def test_image_error_and_contract():
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    b = np.full_like(a, 255)
    assert image_error(a, a)["exact_equal"]
    assert image_error(a, b) == {"mean_abs": 255.0, "max_abs": 255, "exact_equal": False}
    with pytest.raises(ValueError):
        image_error(a, b.astype(np.float32))
