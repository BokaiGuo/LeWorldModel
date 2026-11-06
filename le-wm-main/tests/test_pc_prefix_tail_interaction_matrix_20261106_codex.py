import numpy as np
import pytest

from experiments.pc_prefix_tail_interaction_matrix_dev_20261106_codex import interaction_summary


def test_additive_matrix_has_no_interaction_and_ties():
    matrix = np.arange(4)[:, None] + 2 * np.arange(4)[None, :]
    result = interaction_summary(matrix, matrix, 1e-5)
    assert result["predicted_additive_residual_rms"] == pytest.approx(0.)
    assert result["ties"] == 36


def test_identical_nonadditive_matrices_agree():
    x = np.arange(4)[:, None] * np.arange(4)[None, :]
    result = interaction_summary(x, x, 1e-5)
    assert result["difference_of_differences_agree"] == 36
    assert result["difference_of_differences_disagree"] == 0


def test_wrong_shape_rejected():
    with pytest.raises(ValueError):
        interaction_summary(np.zeros((3, 4)), np.zeros((4, 4)), 1e-5)
