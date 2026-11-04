import numpy as np
import pytest

from experiments.spectral_reachability.run_reacher_pc_feasible_common_suffix import ARMS, fixed_actions


def test_complete_cartesian_action_family_and_invalid_prefix():
    methods = {}
    for i, arm in enumerate(ARMS):
        prefix = np.full((5, 2), i / 10, dtype=np.float32)
        full = np.concatenate([prefix, np.zeros((10, 2), dtype=np.float32)])
        methods[arm] = {"first_action_witness_raw": prefix.tolist(), "action_witness_raw": full.tolist()}
    prefixes, tails = fixed_actions({"methods": methods}, [-1, -1], [1, 1])
    assert len(prefixes) * len(tails) == 30
    methods["selected"]["first_action_witness_raw"][0][0] = 1.0
    with pytest.raises(ValueError):
        fixed_actions({"methods": methods}, [-1, -1], [1, 1])
