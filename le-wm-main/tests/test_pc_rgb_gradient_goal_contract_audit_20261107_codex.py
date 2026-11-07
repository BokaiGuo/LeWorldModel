import numpy as np
import pytest

from experiments.pc_rgb_gradient_goal_contract_audit_20261107_codex import cost_summary, image_difference


def test_image_difference_counts_pixels_not_channels():
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    second = first.copy()
    second[0, 0] = [255, 1, 0]
    assert image_difference(first, second) == {
        "shape": [2, 2, 3], "mae": 256 / 12, "max": 255, "different_pixels": 1}


def test_image_difference_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        image_difference(np.zeros((2, 2, 3)), np.zeros((2, 2, 4)))


def test_cost_summary_uses_matching_target_and_distance():
    p = np.zeros((5, 192), dtype=np.float32)
    a = p.copy()
    p[:, 0] = [3, 2, 1, 4, 5]
    a[:, 0] = [4, 1, 3, 2, 5]
    result = cost_summary(p, a, np.zeros(192), np.array([9, 8, 7, 6, 5]))
    assert result["predicted_index"] == 2
    assert result["actual_oracle_index"] == 1
    assert result["selected_distance"] == 7
    assert result["actual_oracle_distance"] == 8
