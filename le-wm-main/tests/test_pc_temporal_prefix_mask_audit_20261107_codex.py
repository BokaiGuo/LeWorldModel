import numpy as np
import pytest

from experiments.pc_temporal_prefix_mask_audit_20261107_codex import analyze


def test_prefix_minimum_masks_suffix_and_excludes_boundary():
    a = np.array([[[1, 2, 4], [1, 3, 5], [1, 4, 6]]] * 3, dtype=float)
    result = analyze(a, 1)
    assert result["collapsed"] == [True] * 3
    assert result["post_distinguishable"] == [True] * 3
    assert all(all(x) for x in result["all_min_by_prefix"])


def test_post_prefix_can_alter_retention():
    a = np.array([[[0, 10], [0, 11], [0, 12]],
                  [[1, 3], [1, 4], [1, 5]],
                  [[2, 2], [2, 3], [2, 4]]], dtype=float)
    result = analyze(a, 1)
    assert result["full_kept"] == [0, 1]
    assert result["post_kept"] == [2, 1]
    assert result["set_changed"]


def test_invalid_input_rejected():
    with pytest.raises(ValueError):
        analyze(np.zeros((3, 3, 2)), 2)
    with pytest.raises(ValueError):
        analyze(np.full((3, 3, 2), np.nan), 1)
