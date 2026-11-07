import numpy as np
import pytest

from experiments.pc_tworoom_nonlinear_selection_20261107_codex import select_candidate


def test_select_decoded_goal_ties_by_latent_cost_then_index():
    offsets = [[8, 0], [0, 0], [0, 0], [0, 0]]
    assert select_candidate(offsets, [0, 3, 2, 2]) == 2


def test_select_rejects_nonfinite_or_mismatched():
    with pytest.raises(ValueError):
        select_candidate([[0, 0]], [float("nan")])
    with pytest.raises(ValueError):
        select_candidate(np.zeros((2, 2)), [1])
