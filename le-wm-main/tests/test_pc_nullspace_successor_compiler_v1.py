import numpy as np

from experiments.spectral_reachability import pc_nullspace_successor_compiler_v1 as compiler


def test_null_directions_orthogonal_to_response_and_each_other():
    rng = np.random.default_rng(23)
    basis = np.linalg.qr(rng.normal(size=(10, 2)))[0].T
    directions, attempts = compiler.extra_directions(np.random.default_rng(7), basis, "null")
    np.testing.assert_allclose(directions @ basis.T, 0, atol=1e-12)
    np.testing.assert_allclose(directions @ directions.T, np.eye(2), atol=1e-12)
    assert attempts == [1, 1]


def test_zero_rank_still_explores_and_controls_share_random_stream():
    null, _ = compiler.extra_directions(np.random.default_rng(8), [], "null")
    random, _ = compiler.extra_directions(np.random.default_rng(8), [], "random_extra")
    np.testing.assert_allclose(null, random, atol=1e-12)
    np.testing.assert_allclose(null @ null.T, np.eye(2), atol=1e-12)
