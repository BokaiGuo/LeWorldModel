import numpy as np
import pytest

from experiments.pc_local_perturb_effect_shuffle_correction_20261107_codex import effect_scores


def test_matched_and_shuffled_keep_own_parent_baseline():
    parents = np.arange(8, dtype=float) * 100
    delta = np.repeat(np.arange(8, dtype=float)[:, None], 4, axis=1)
    mapping = np.roll(np.arange(8), 1)
    matched, shuffled = effect_scores(parents, parents[:, None] + delta, mapping)
    np.testing.assert_allclose(matched, parents + delta[:, 0] * 4 / 5)
    np.testing.assert_allclose(shuffled, parents + delta[mapping, 0] * 4 / 5)
    offset = np.arange(8, dtype=float) * 71
    _, shifted = effect_scores(parents + offset, parents[:, None] + delta + offset[:, None], mapping)
    np.testing.assert_allclose(shifted - shuffled, offset)


def test_rejects_non_derangement():
    with pytest.raises(ValueError):
        effect_scores(np.zeros(8), np.zeros((8, 4)), np.arange(8))
