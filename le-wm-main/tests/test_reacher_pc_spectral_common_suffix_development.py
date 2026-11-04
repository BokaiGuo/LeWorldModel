import numpy as np
import pytest

from experiments.spectral_reachability.run_reacher_pc_spectral_common_suffix_development import (
    ARMS, fixed_actions,
)


def test_fixed_actions_preserves_all_tails_and_rejects_misalignment():
    methods = {}
    for index, arm in enumerate(ARMS):
        full = np.full((15, 2), index / 10, dtype=np.float32)
        methods[arm] = {"first_action_witness_raw": full[:5].tolist(), "action_witness_raw": full.tolist()}
    prefixes, tails = fixed_actions({"methods": methods}, np.array([-1, -1]), np.array([1, 1]))
    assert set(prefixes) == set(ARMS)
    assert set(tails) == {"passive_zero", "selected_tail", "zero_tail", "same_bank_greedy_tail"}
    assert np.array_equal(tails["passive_zero"], np.zeros((10, 2)))
    assert np.allclose(tails["same_bank_greedy_tail"], 0.2)
    methods["selected"]["first_action_witness_raw"][0][0] = 0.5
    with pytest.raises(ValueError, match="unbound"):
        fixed_actions({"methods": methods}, np.array([-1, -1]), np.array([1, 1]))
