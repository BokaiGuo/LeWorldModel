import numpy as np
import pytest

from experiments.spectral_reachability.pc_atlas_matched_exact_dev_20261103_exclusive import ARMS, run_arm


def test_matched_call_counts_bounds_and_shared_candidates():
    rows = [run_arm(2, 42, arm) for arm in ARMS]
    for row in rows:
        assert row["counts"] == {"rollout_attempted": 28, "rollout_returned": 28,
                                 "jacobian_attempted": 5, "jacobian_returned": 5}
        assert row["backend_action_steps"] == {"rollout": 32, "jacobian": 5}
        assert row["candidates"][row["selected"]]["terminal_cost"] == min(
            c["terminal_cost"] for c in row["candidates"])
        for c in row["candidates"]:
            actions = np.asarray(c["actions"])
            assert actions.shape == (2, 2)
            assert np.all(np.abs(actions[:, 0]) <= 1.)
            assert np.all(np.abs(actions[:, 1]) <= np.pi / 2)
    assert rows[0]["candidates"][:2] == rows[1]["candidates"][:2] == rows[2]["candidates"][:2]


def test_reproducibility_and_validation():
    assert run_arm(0, 42, "random") == run_arm(0, 42, "random")
    assert run_arm(0, 42, "random")["candidates"][2:] != run_arm(0, 43, "random")["candidates"][2:]
    with pytest.raises(ValueError):
        run_arm(64, 42, "random")
