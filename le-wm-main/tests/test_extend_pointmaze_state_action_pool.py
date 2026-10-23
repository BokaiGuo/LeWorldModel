from __future__ import annotations

import numpy as np
import pytest

from experiments.decision_benchmark.extend_pointmaze_state_action_pool import rollout_trajectory


def test_extension_requires_enough_new_roots_for_the_isolated_protocol(tmp_path) -> None:
    from experiments.decision_benchmark.extend_pointmaze_state_action_pool import build_extension
    with pytest.raises(ValueError, match="at least 208"):
        build_extension(tmp_path, tmp_path / "out", episode_count=207)


def test_rollout_rejects_invalid_action_shape_before_touching_mujoco() -> None:
    with pytest.raises(ValueError, match="actions must be finite"):
        rollout_trajectory(np.zeros(4), np.zeros((100, 3)))
