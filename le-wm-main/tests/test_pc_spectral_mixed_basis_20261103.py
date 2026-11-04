import numpy as np
import pytest

from experiments.spectral_reachability.pc_spectral_mixed_basis_20261103 import (
    SEEDS, disk_coefficients, run_arm,
)


def test_coefficients_are_deterministic_and_in_unit_disk():
    first = disk_coefficients(SEEDS[0])
    np.testing.assert_array_equal(first, disk_coefficients(SEEDS[0]))
    assert first.shape == (16, 2)
    assert np.all(np.linalg.norm(first, axis=1) <= 1.)
    assert np.all(np.abs(first) > 0.)


def test_actions_optimizer_controls_and_charged_jacobian():
    arms = [run_arm(0, SEEDS[0], source) for source in
            ("raw_isotropic", "spectral_isotropic", "spectral_anisotropic")]
    assert all(len(arm["candidates"]) == 18 for arm in arms)
    for i in (0, 1):
        assert arms[0]["candidates"][i] == arms[1]["candidates"][i] == arms[2]["candidates"][i]
    assert arms[0]["candidates"] == arms[1]["candidates"]
    assert arms[0]["operation_counts"]["rollout_attempted"] == 112
    assert arms[0]["operation_counts"]["jacobian_attempted"] == 19
    assert arms[1]["operation_counts"]["jacobian_attempted"] == 20
    for arm in arms:
        for candidate in arm["candidates"]:
            actions = np.asarray(candidate["actions"])
            assert actions.shape == (2, 2)
            assert np.max(np.abs(actions[:, 0])) <= 1.
            assert np.max(np.abs(actions[:, 1])) <= np.pi / 2
            assert candidate["terminal_cost"] == pytest.approx(candidate["terminal_position_error"]**2 / 3)
        assert arm["candidates"][arm["selected_index"]]["terminal_cost"] == min(
            row["terminal_cost"] for row in arm["candidates"])


def test_refuses_unregistered_development_cases():
    with pytest.raises(ValueError):
        run_arm(100, 101, "raw_isotropic")
