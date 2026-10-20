import numpy as np

from experiments.audit_pusht_verified_orientation_linesearch import signed_contact_action


def test_signed_contact_action_is_bounded_and_has_inward_component():
    class FakeEnv:
        pass

    # The helper needs the real geometry, so test its output contract with a
    # tiny monkeypatch at the module boundary rather than constructing a full
    # MuJoCo environment in a unit test.
    import experiments.audit_pusht_verified_orientation_linesearch as module

    module.boundary_inward_normal = lambda env, state: np.asarray([1.0, 0.0])
    action = signed_contact_action(FakeEnv(), np.zeros(7), 1.0, 0.1, 0.16)
    assert np.all(np.abs(action) <= 1.0)
    assert action[0] > 0.0
    assert action[1] > 0.0


def test_signed_contact_action_reverses_only_tangent_component():
    import experiments.audit_pusht_verified_orientation_linesearch as module

    module.boundary_inward_normal = lambda env, state: np.asarray([0.0, 1.0])
    positive = signed_contact_action(None, np.zeros(7), 1.0, 0.1, 0.16)
    negative = signed_contact_action(None, np.zeros(7), -1.0, 0.1, 0.16)
    assert np.allclose(positive[1], negative[1])
    assert np.isclose(positive[0], -negative[0])
