import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_multistart_cp5_20261030_main as mod


def test_fixed_start_roster_and_bounds():
    starts = mod.initial_actions(2)
    assert starts.shape == (4, 2, 5, 2)
    assert np.array_equal(starts, mod.initial_actions(2))
    assert np.count_nonzero(starts[0]) == 0
    assert np.all(np.abs(starts) <= 1)
    assert not np.array_equal(starts, mod.initial_actions(3))
    with pytest.raises(ValueError):
        mod.initial_actions(8)


def test_selection_tie_and_finite_check():
    assert mod.pick([2.0, 1.0, 1.0, 3.0]) == 1
    with pytest.raises(ValueError):
        mod.pick([1.0, np.nan, 2.0, 3.0])
