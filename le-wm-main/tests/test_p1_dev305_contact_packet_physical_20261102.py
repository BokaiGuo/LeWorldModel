import numpy as np
import pytest

from experiments.spectral_reachability.p1_dev305_contact_packet_physical_20261102 import candidate_packets


def test_packets_are_bounded_and_symmetric():
    actions = candidate_packets([.6, .8])
    assert set(actions) == {"toward", "away", "zero"}
    np.testing.assert_array_equal(actions["toward"], -actions["away"])
    np.testing.assert_array_equal(actions["zero"], np.zeros((5, 2)))
    assert np.max(np.abs(actions["toward"])) <= .5


@pytest.mark.parametrize("direction", [[0, 0], [2, 0], [float("nan"), 1], [1]])
def test_invalid_direction_rejected(direction):
    with pytest.raises(ValueError):
        candidate_packets(direction)
