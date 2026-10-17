from __future__ import annotations

import numpy as np

from experiments.run_live_cube_rgb_macro_ranker import (
    _approach_observation,
    _rgb_approach_action,
)


def test_rgb_approach_action_is_bounded_and_has_five_controls() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:12, 24:28, 0] = 220
    image[8:12, 24:28, 1] = 20
    image[8:12, 24:28, 2] = 20
    # The frozen RGB rule must remain a bounded action even with incomplete
    # effector visibility.
    action = _rgb_approach_action(image)
    assert action.shape == (5,)
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_approach_observation_marks_missing_effector_without_fabricating_error() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:12, 24:28, 0] = 220
    observation = _approach_observation(image)
    assert observation["block_valid"] is True
    assert observation["effector_valid"] is False
    assert observation["approach_error_norm"] is None
