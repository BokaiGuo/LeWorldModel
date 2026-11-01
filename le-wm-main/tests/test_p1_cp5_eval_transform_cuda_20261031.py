import numpy as np
import pytest

from experiments.spectral_reachability.p1_cp5_eval_transform_cuda_20261031 import compare_arrays


def test_comparison_detects_bitwise_and_numeric_difference():
    a = np.array([0.0, 1.0], dtype=np.float32)
    b = np.array([-0.0, 1.25], dtype=np.float32)
    assert compare_arrays(a, a)["exact_bytes"] is True
    result = compare_arrays(a, b)
    assert result["exact_bytes"] is False
    assert result["max_abs_difference"] == 0.25
    assert result["left_sha256"] != result["right_sha256"]


def test_comparison_rejects_shape_dtype_and_nonfinite():
    a = np.array([1.0], dtype=np.float32)
    for invalid in (a.astype(np.float64), np.array([[1.0]], np.float32),
                    np.array([float("nan")], np.float32)):
        with pytest.raises(ValueError):
            compare_arrays(a, invalid)
