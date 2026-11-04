import numpy as np
import pytest

from experiments.spectral_reachability.p1_dev305_contact_equal_scale_20261102 import choose_action


def test_same_scale_and_different_direction():
    feedback = choose_action("rgb_feedback", [0, 1], [1, 0])
    fixed = choose_action("fixed", [0, 1], [1, 0])
    conservative = choose_action("guard16", [0, 1], [1, 0])
    np.testing.assert_allclose(np.linalg.norm(feedback), np.linalg.norm(fixed))
    np.testing.assert_allclose(feedback, conservative)
    np.testing.assert_allclose(choose_action("zero", [0, 1], [1, 0]), [0, 0])


@pytest.mark.parametrize("mode,direction", [("fixed", [0, 0]), ("bad", [1, 0])])
def test_invalid(mode, direction):
    with pytest.raises(ValueError):
        choose_action(mode, [1, 0], direction)
