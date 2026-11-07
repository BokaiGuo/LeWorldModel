import numpy as np
import pytest

from experiments.pc_pointmaze_render_drift_diagnostic_20261107_codex import metrics


def test_metrics_distinguishes_encoding_and_decision():
    old = np.array([[0.], [2.]])
    new = np.array([[3.], [1.]])
    result = metrics(old, new, [0.], [2., 1.])
    assert result["old_choice"] == 0
    assert result["new_choice"] == 1
    assert result["old_physical_regret"] == 1.
    assert result["new_physical_regret"] == 0.


def test_metrics_rejects_nonfinite():
    with pytest.raises(ValueError):
        metrics([[0], [1]], [[0], [1]], [0], [0, float("nan")])
