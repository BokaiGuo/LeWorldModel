import numpy as np

from experiments.audit_pusht_orientation_response import matched_actions, wrap_angle


def test_wrap_angle_keeps_small_paired_difference_continuous():
    assert abs(float(wrap_angle(2.0 * np.pi + 0.2)) - 0.2) < 1e-8


def test_matched_actions_have_equal_nonzero_norms_and_opposite_moments():
    chart = {
        "contact_point_world": np.asarray([2.0, 0.0]),
        "contact_radius": np.asarray([2.0, 0.0]),
        "inward": np.asarray([1.0, 0.0]),
        "radial_tangent": np.asarray([0.0, 1.0]),
    }
    actions = matched_actions(chart, inward_magnitude=0.2, tangent_magnitude=0.5)
    assert abs(np.linalg.norm(actions["plus"]) - np.linalg.norm(actions["minus"])) < 1e-8
    assert actions["pair_valid"]
    assert np.sign(actions["moment_plus"]) == -np.sign(actions["moment_minus"])


def test_zero_pulse_is_the_common_drift_reference():
    chart = {
        "contact_point_world": np.asarray([2.0, 0.0]),
        "contact_radius": np.asarray([2.0, 0.0]),
        "inward": np.asarray([1.0, 0.0]),
        "radial_tangent": np.asarray([0.0, 1.0]),
    }
    actions = matched_actions(chart, inward_magnitude=0.2, tangent_magnitude=0.5)
    assert np.allclose(actions["zero"], np.zeros(2))
