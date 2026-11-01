import numpy as np

from experiments.spectral_reachability.calibrate_reacher_d2p_qpos_readout import decode_qpos, fit_ridge_angle_readout, wrapped_delta


def test_frozen_latent_ridge_readout_recovers_two_angular_coordinates():
    rng = np.random.default_rng(7)
    qpos = rng.uniform(-1.0, 1.0, size=(160, 2))
    latents = np.concatenate((np.sin(qpos), np.cos(qpos), rng.normal(size=(160, 1))), axis=1)
    artifact = fit_ridge_angle_readout(latents, qpos, alpha=1.0e-6)
    prediction = decode_qpos(artifact, latents)
    assert np.abs(wrapped_delta(prediction, qpos)).mean() < 1.0e-4


def test_wrapped_delta_respects_the_pi_boundary():
    delta = wrapped_delta(np.asarray([[-np.pi + 0.01, 0.0]]), np.asarray([[np.pi - 0.01, 0.0]]))
    np.testing.assert_allclose(delta, np.asarray([[0.02, 0.0]]), atol=1.0e-6)
