import numpy as np

from experiments.run_pusht_contact_history_velocity_diagnostic import (
    contact_geometry_features,
    feature_vector,
    mode_for_state,
)


def test_object_centric_contact_terms_are_translation_invariant():
    state = np.array([100.0, 120.0, 145.0, 160.0, 0.4, 12.0, -8.0])
    shifted = state.copy()
    shifted[:4] += np.array([37.0, -19.0, 37.0, -19.0])
    action = np.array([0.35, -0.25])
    # The first two fields retain absolute block position because PushT has
    # walls; all relative/contact terms must remain unchanged.
    assert np.allclose(
        contact_geometry_features(state, action)[2:],
        contact_geometry_features(shifted, action)[2:],
    )


def test_history_features_add_previous_action_and_increment():
    state = np.array([100.0, 120.0, 145.0, 160.0, 0.4, 12.0, -8.0])
    action = np.array([0.35, -0.25])
    previous = np.array([-0.1, 0.2])
    features = feature_vector(state, action, previous, "history")
    assert features.shape == (14,)
    assert np.allclose(features[-4:-2], previous)
    assert np.allclose(features[-2:], action - previous)
    assert np.isfinite(features).all()


def test_contact_mode_boundaries_are_explicit():
    free = np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
    approach = np.array([0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 0.0])
    contact = np.array([0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0])
    assert mode_for_state(free) == "free"
    assert mode_for_state(approach) == "approach"
    assert mode_for_state(contact) == "contact"
