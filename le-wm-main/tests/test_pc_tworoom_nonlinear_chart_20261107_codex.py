import numpy as np
import pytest

from experiments.pc_tworoom_nonlinear_chart_20261107_codex import nearest, templates


def test_templates_preserve_zero_and_reject_cropped_marker():
    current = np.full((224, 224, 3), 100, dtype=np.uint8)
    goal = current.copy()
    current[30:42, 30:42] = [255, 0, 0]
    goal[100:112, 100:112] = [255, 0, 0]
    images, offsets, info = templates(current, goal, offsets=[(0, 0), (8, 0), (150, 0)])
    assert info["valid"]
    assert offsets.tolist() == [[0, 0], [8, 0]]
    np.testing.assert_array_equal(images[0], goal)
    assert np.array_equal(images[1][100:112, 108:120], np.full((12, 12, 3), [255, 0, 0]))


def test_nearest_uses_order_for_ties_and_checks_finiteness():
    offsets = np.array([[0, 0], [8, 0]])
    np.testing.assert_array_equal(nearest([[0., 0.], [2., 0.]], [[1., 0.]], offsets), [[0, 0]])
    with pytest.raises(ValueError):
        nearest([[0., float("nan")]], [[0., 0.]], offsets[:1])
