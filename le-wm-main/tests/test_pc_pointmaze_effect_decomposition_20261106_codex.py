import numpy as np
import pytest

from experiments.pc_pointmaze_effect_decomposition_20261106_codex import decompose


def test_identical_predictions_recover_effect_and_goal_ranking():
    z = np.array([[0., 0.], [1., 0.], [2., 0.]])
    result = decompose(z, z, np.array([2., 0.]), [2., 1., 0.])
    assert result["choices"] == {"predicted": 2, "real_encoded": 2, "physical": 2}
    assert all(p["predicted_vs_real_effect_cosine"] == pytest.approx(1) for p in result["pairs"])
    assert all(p["effect_relative_l2_error"] == pytest.approx(0) for p in result["pairs"])


def test_common_residual_preserves_effect_but_can_change_choice():
    z = np.array([[0.], [1.], [2.]])
    before = decompose(z, z, np.array([2.]), [2., 1., 0.])
    after = decompose(z - 3, z, np.array([2.]), [2., 1., 0.])
    assert [p["effect_relative_l2_error"] for p in before["pairs"]] == pytest.approx(
        [p["effect_relative_l2_error"] for p in after["pairs"]])
    assert after["choices"]["predicted"] == 2
    shifted = decompose(z + 3, z, np.array([2.]), [2., 1., 0.])
    assert shifted["choices"]["predicted"] == 0


def test_rejects_shape_and_nonfinite():
    with pytest.raises(ValueError):
        decompose(np.zeros((2, 1)), np.zeros((3, 1)), [0], [0, 0, 0])
    with pytest.raises(ValueError):
        decompose(np.full((3, 1), np.nan), np.zeros((3, 1)), [0], [0, 0, 0])
