import numpy as np

from experiments.spectral_reachability.pc_rotated_spectral_basis_dev_20261103 import (
    REFERENCE, run_arm, samples,
)


def test_samples_are_deterministic_and_bounded():
    a = samples(101)
    assert a.shape == (16, 4)
    np.testing.assert_array_equal(a, samples(101))
    assert np.max(np.linalg.norm(a, axis=1)) <= .35


def test_rotated_basis_and_matched_backend_counts():
    arms = [run_arm(0, 101, source) for source in
            ("raw_isotropic", "spectral_isotropic", "spectral_anisotropic")]
    basis = np.array(arms[1]["basis"])
    np.testing.assert_allclose(basis @ basis.T, np.eye(4), atol=1e-12)
    assert not np.allclose(np.abs(basis), np.eye(4))
    assert all(a["operation_counts"] == arms[0]["operation_counts"] for a in arms)
    assert all(len(a["candidates"]) == 18 for a in arms)
    assert REFERENCE.shape == (2, 2)
