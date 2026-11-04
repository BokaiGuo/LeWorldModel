import numpy as np
import pytest

from experiments.run_pusht_counterfactual_aligned_20261102 import (
    PUBLIC_ACTION_MEAN,
    PUBLIC_ACTION_STD,
    normalize_public_action,
    normalized_action_sequence,
    rollout,
)


def test_live_command_is_not_divided_by_internal_environment_scale():
    command = np.array([0.5, -0.25], dtype=np.float32)
    np.testing.assert_allclose(normalize_public_action(command),
                               (command - PUBLIC_ACTION_MEAN) / PUBLIC_ACTION_STD)
    assert not np.allclose(normalize_public_action(command),
                           (command / 100 - PUBLIC_ACTION_MEAN) / PUBLIC_ACTION_STD)


def test_model_packet_count_matches_physical_horizon():
    command = np.array([0.5, -0.25], dtype=np.float32)
    result = normalized_action_sequence(command, 10)
    assert result.shape == (2, 10)
    np.testing.assert_array_equal(result[0], np.tile(normalize_public_action(command), 5))
    np.testing.assert_array_equal(result[1], result[0])
    for invalid in (0, 1, 8, 11):
        with pytest.raises(ValueError):
            normalized_action_sequence(command, invalid)


def test_rollout_reports_native_censoring():
    class Env:
        unwrapped = None

        def __init__(self):
            self.unwrapped = self
            self.steps = 0

        def reset(self, **_kwargs):
            return {}, {}

        def step(self, _action):
            self.steps += 1
            return {}, 0, self.steps == 3, False, {"goal_state": np.zeros(7)}

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def _get_obs(self):
            return np.zeros(7)

    result = rollout(Env(), np.zeros(7), np.zeros(7), np.zeros(2, np.float32), 10, 1)
    assert result["executed_steps"] == 3
    assert result["terminated"] and not result["truncated"]
