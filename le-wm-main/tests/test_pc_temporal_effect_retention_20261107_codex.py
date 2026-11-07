import numpy as np
import pytest

from experiments.pc_temporal_effect_retention_20261107_codex import choose, retention


def test_prefix_score_does_not_mask_retention_and_shuffle_is_reproducible():
    rng = np.random.default_rng(7)
    scores = rng.normal(size=(8, 9, 5))
    scores[:, :, :2] = -1e9
    result = retention(scores, 9)
    assert result == retention(scores, 9)
    assert all(len(set(v)) == 2 for v in result.values())
    assert result["lookahead"][0] == result["gated"][0]


def test_selection_never_uses_unavailable_b_and_includes_incumbent():
    a = np.full((8, 8), 4.0)
    ids = np.arange(16)
    b = np.full(16, 3.0)
    assert choose(a, b, ids, [0, 1], np.array([1.0, 2.0]))[1:] == ("incumbent", 0)
    with pytest.raises(ValueError, match="missing B"):
        choose(a, b, ids, [0, 2], np.array([1.0, 2.0]))
