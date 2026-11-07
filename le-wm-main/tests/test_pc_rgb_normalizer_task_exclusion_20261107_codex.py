import numpy as np
import pytest
from experiments.pc_rgb_normalizer_task_exclusion_20261107_codex import fit_scalers


def test_exclusion_applies_to_entire_episode_and_nan_rows():
    actions = np.array([[1., 2.], [3., 4.], [5., 6.], [np.nan, 1.]])
    full, reduced, counts = fit_scalers(actions, np.array([1, 1, 2, 2]), [1])
    assert counts == {1: 2}
    np.testing.assert_allclose(full.mean_, [3, 4])
    np.testing.assert_allclose(reduced.mean_, [5, 6])
    assert int(reduced.n_samples_seen_) == 1


def test_reject_missing_or_all_excluded():
    with pytest.raises(ValueError):
        fit_scalers(np.ones((2, 2)), [1, 1], [1])
    with pytest.raises(ValueError):
        fit_scalers(np.ones((2, 2)), [1, 1], [2])
