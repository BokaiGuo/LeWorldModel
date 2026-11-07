import numpy as np
import pytest

from experiments.pc_pointmaze_heldout_amplitude_encoding_20261107_codex import comparisons


def test_decomposition_tracks_model_and_encoded_disagreement():
    predicted = np.array([[0.], [1.], [2.], [3.]])
    actual = np.array([[2.], [0.], [1.], [3.]])
    result = comparisons(predicted, actual, np.array([0.]), np.array([3., 1., 2., 4.]))
    assert result["selected"] == {"predicted": 0, "encoded": 1, "physical": 1}
    assert result["regret"]["predicted"] == 2
    assert result["pairs"][0]["prediction_encoding"] is False


def test_decomposition_rejects_nonfinite():
    with pytest.raises(ValueError):
        comparisons(np.zeros((4, 1)), np.zeros((4, 1)), [0], [1, 2, 3, float("nan")])
