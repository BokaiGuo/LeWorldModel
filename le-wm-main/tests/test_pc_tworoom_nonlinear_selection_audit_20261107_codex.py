import numpy as np
import pytest

from experiments.pc_tworoom_nonlinear_selection_audit_20261107_codex import (
    image_difference, reconstruct_choice,
)


def test_independent_reconstruction_and_stable_ties():
    embeddings = np.array([[0., 0.], [1., 0.], [0., 1.]])
    forecasts = np.array([[1., 0.], [0., 1.], [0., 1.]])
    offsets = np.array([[3, 3], [0, 0], [0, 0]])
    index, decoded = reconstruct_choice(embeddings, forecasts, offsets, [3, 2, 2])
    assert index == 1
    np.testing.assert_array_equal(decoded, [[0, 0], [0, 0], [0, 0]])


def test_invalid_input_and_image_mismatch_fail_closed():
    with pytest.raises(ValueError):
        reconstruct_choice([[0]], [[float("nan")]], [[0, 0]], [1])
    with pytest.raises(ValueError):
        image_difference(np.zeros((2, 2)), np.zeros((3, 2)))
    assert image_difference(np.array([0], dtype=np.uint8), np.array([255], dtype=np.uint8))["max"] == 255
