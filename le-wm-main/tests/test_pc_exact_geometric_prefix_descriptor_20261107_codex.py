import numpy as np
import pytest

from experiments.pc_exact_geometric_prefix_descriptor_20261107_codex import select_with_incumbent


def test_only_selected_refinement_can_beat_shared_incumbent():
    pool = np.zeros((8, 15, 2))
    refinements = [{"best": 0, "costs": [0.001 if i == 7 else 4.0], "actions": [pool[i].tolist()]}
                   for i in range(8)]
    result = select_with_incumbent(pool, np.arange(8) + 1., refinements, [1, 2])
    assert result["source"] == "incumbent"
    assert result["score"] == 1
    assert select_with_incumbent(pool, np.arange(8) + 1., refinements, [2, 7])["score"] == .001


def test_reject_duplicate_or_illegal_slots():
    with pytest.raises(ValueError):
        select_with_incumbent(np.zeros((8, 15, 2)), np.zeros(8), [], [1, 1])
    with pytest.raises(ValueError):
        select_with_incumbent(np.zeros((8, 15, 2)), np.zeros(8), [], [0, 8])
