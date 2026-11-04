import numpy as np
import pytest

from experiments.spectral_reachability.p1_dev305_contact_rgb_feedback_20261102 import choose_action


def test_feedback_changes_direction_and_controls_are_fixed():
    np.testing.assert_allclose(choose_action("rgb_feedback", [0, 1], [1, 0]), [0, .2])
    np.testing.assert_allclose(choose_action("fixed", [0, 1], [1, 0]), [.5, 0])
    np.testing.assert_allclose(choose_action("zero", [0, 1], [1, 0]), [0, 0])


@pytest.mark.parametrize("mode,direction", [("rgb_feedback", [0, 0]), ("wrong", [1, 0]), ("rgb_feedback", [float("nan"), 0])])
def test_bad_policy_or_direction_rejected(mode, direction):
    with pytest.raises(ValueError):
        choose_action(mode, direction, [1, 0])
