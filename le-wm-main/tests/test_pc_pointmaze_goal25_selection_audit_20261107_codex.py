import numpy as np
import pytest

from experiments.pc_pointmaze_goal25_selection_audit_20261107_codex import reconstruct_cem


def test_reconstruction_is_deterministic_and_bounded():
    record = {"seed": 17, "elite_count": 4, "round_costs": [list(range(16)) for _ in range(3)]}
    a, costs = reconstruct_cem(record)
    b, _ = reconstruct_cem(record)
    assert a.shape == (48, 25, 2)
    assert costs.shape == (48,)
    assert np.abs(a).max() <= 1
    np.testing.assert_array_equal(a, b)


def test_reconstruction_rejects_incomplete_costs():
    with pytest.raises(ValueError):
        reconstruct_cem({"seed": 17, "elite_count": 4, "round_costs": [list(range(16))]})
