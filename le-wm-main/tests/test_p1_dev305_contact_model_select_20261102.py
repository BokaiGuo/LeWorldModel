import numpy as np
import pytest

from experiments.spectral_reachability.p1_dev305_contact_model_select_20261102 import packets


def test_fixed_bank_and_raw_units():
    bank = packets([.6, .8])
    assert bank.shape == (4, 5, 2)
    np.testing.assert_array_equal(bank[0], -bank[1])
    np.testing.assert_array_equal(bank[2], 0)
    np.testing.assert_allclose(np.linalg.norm(bank[3, 0]), .5)


@pytest.mark.parametrize("value", [[0, 0], [2, 0], [float("nan"), 0]])
def test_invalid_direction(value):
    with pytest.raises(ValueError):
        packets(value)
