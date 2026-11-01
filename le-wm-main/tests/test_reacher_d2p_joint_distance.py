import numpy as np
import pytest

from experiments.spectral_reachability.reacher_d2p_joint_distance import distance
from experiments.spectral_reachability.analyze_reacher_d2p_masked_feedback import joint_distance


def test_bounded_wrist_must_not_wrap_but_shoulder_must():
    assert distance([2.7, 2.7], [-2.7, -2.7], [False, True]) == pytest.approx(np.hypot(2*np.pi-5.4, 5.4))
    assert distance([0., 2.7], [0., -2.7], [False, True]) == pytest.approx(5.4)


def test_independent_implementation_agrees_full_random_states():
    rng = np.random.default_rng(10)
    for _ in range(50):
        a, b = rng.normal(size=(2, 2))*4
        assert distance(a, b, [False, True]) == pytest.approx(joint_distance(a, b, [False, True]), abs=1e-14)


@pytest.mark.parametrize("a,b,limits", [([0.], [0., 0.], [False, True]),
    ([0., float("nan")], [0., 0.], [False, True]), ([0., 0.], [0., 0.], [0, 1])])
def test_invalid_joint_contracts_rejected(a, b, limits):
    with pytest.raises(ValueError): distance(a, b, limits)
