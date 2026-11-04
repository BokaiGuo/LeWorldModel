import numpy as np
import pytest

from experiments.spectral_reachability.pc_atlas_matched_nullspace_dev_20261103 import run_arm


def test_nullspace_bank_and_actual_backend_counts():
    row = run_arm(3, 42)
    np.testing.assert_allclose(np.abs(row["root_null_direction"]), [0, 1], atol=1e-12)
    assert row["counts"]["rollout_returned"] == 28
    assert row["counts"]["jacobian_returned"] == 5
    assert row["backend_action_steps"] == {"rollout": 32, "jacobian": 5}
    prefixes = np.array([item["actions"][0] for item in row["candidates"]])
    np.testing.assert_allclose(prefixes[2:], [[0, -.5], [0, .5]], atol=1e-12)


def test_rejects_unregistered_goal():
    with pytest.raises(ValueError):
        run_arm(8, 42)
