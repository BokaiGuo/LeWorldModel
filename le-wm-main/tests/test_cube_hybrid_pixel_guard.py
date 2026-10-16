import numpy as np

from experiments.run_cube_hybrid_contact_latent import (
    _fit_online_pixel_action_mapping,
    _pixel_guard_should_fallback,
)


def test_pixel_guard_triggers_only_after_fixed_error_margin():
    assert not _pixel_guard_should_fallback(None, 4.0, False, 0.5)
    assert not _pixel_guard_should_fallback(4.0, 4.5, False, 0.5)
    assert _pixel_guard_should_fallback(4.0, 4.51, False, 0.5)


def test_pixel_guard_is_irreversible_after_trigger():
    assert _pixel_guard_should_fallback(9.0, 0.0, True, 0.5)


def test_pixel_guard_rejects_negative_threshold():
    try:
        _pixel_guard_should_fallback(1.0, 2.0, False, -0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative pixel guard threshold must be rejected")


def test_online_pixel_mapping_recovers_local_jacobian_with_ridge_prior():
    prior = np.eye(2, dtype=np.float32)
    actions = [
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([-1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, -1.0], dtype=np.float32),
    ]
    true_mapping = np.asarray([[0.1, 0.3], [0.25, 0.05]], dtype=np.float32)
    deltas = [action @ true_mapping for action in actions]
    fitted = _fit_online_pixel_action_mapping(prior, actions, deltas, ridge=1.0e-6)
    np.testing.assert_allclose(fitted, true_mapping, atol=1.0e-4)


def test_online_pixel_mapping_keeps_prior_before_two_action_directions():
    prior = np.asarray([[0.1, 0.3], [0.25, 0.05]], dtype=np.float32)
    fitted = _fit_online_pixel_action_mapping(
        prior,
        [np.asarray([1.0, 0.0], dtype=np.float32)],
        [np.asarray([0.1, 0.3], dtype=np.float32)],
        ridge=0.01,
    )
    np.testing.assert_array_equal(fitted, prior)
