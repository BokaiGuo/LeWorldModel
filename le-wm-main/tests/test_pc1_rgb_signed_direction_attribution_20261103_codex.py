import numpy as np
import pytest

from experiments.pc1_rgb_signed_direction_attribution_20261103_codex import analyze, stats


def test_even_curvature_is_invisible_to_central_difference():
    # f(t) = t + 2 t^2, in one output coordinate.
    item = {"scale": "candidate", "epsilon": 0.5, "plus": [1.0, 0.0], "minus": [0.0, 0.0]}
    row = {"root": 0, "branch": "first", "direction_index": 0, "center": [0.0, 0.0],
           "ad": [1.0, 0.0], "saved_jv": [1.0, 0.0], "measurements": [item] * 3}
    rows = analyze({"rows": [row] * 24})
    assert rows[0]["even_norm"] == pytest.approx(0.5)
    assert rows[0]["odd_norm"] == pytest.approx(0.5)
    assert rows[0]["signs"][0]["absolute_error"]["ad"] == pytest.approx(0.5)
    assert rows[0]["signs"][1]["relative_error"]["ad"] is None
    assert np.allclose(rows[0]["even"], [0.5, 0.0])


def test_stats_preserves_missing_and_reports_empty():
    assert stats([None]) == {"n": 0, "median": None, "max": None}
    assert stats([1, None, 3]) == {"n": 2, "median": 2.0, "max": 3.0}
