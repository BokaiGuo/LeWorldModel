from __future__ import annotations

import numpy as np

from experiments.decision_benchmark import build_pointmaze_velocity_matched_state_pool as pool


def test_velocity_match_requires_close_position_and_opposed_nontrivial_velocity() -> None:
    plus = np.zeros((100, 4), dtype=np.float32)
    minus = np.zeros((100, 4), dtype=np.float32)
    plus[pool.SOURCE_FRAME] = [0.2, -0.1, 0.3, 0.0]
    minus[pool.SOURCE_FRAME] = [0.21, -0.1, -0.3, 0.0]
    assert pool._pair_is_matched(plus, minus, position_tolerance=0.025, minimum_speed=0.08)
    minus[pool.SOURCE_FRAME, 2] = 0.3
    assert not pool._pair_is_matched(plus, minus, position_tolerance=0.025, minimum_speed=0.08)


def test_velocity_match_requires_declared_target_and_visible_history_when_requested() -> None:
    plus = np.zeros((100, 4), dtype=np.float32)
    minus = np.zeros((100, 4), dtype=np.float32)
    plus[: pool.SOURCE_FRAME + 1, :2] = [0.4, 0.5]
    minus[: pool.SOURCE_FRAME + 1, :2] = [0.4, 0.5]
    plus[pool.SOURCE_FRAME, 2:] = [0.3, 0.0]
    minus[pool.SOURCE_FRAME, 2:] = [-0.3, 0.0]
    assert pool._pair_is_matched(
        plus, minus, position_tolerance=0.025, minimum_speed=0.08,
        target_qpos=np.asarray([0.4, 0.5]), history_qpos_bounds=(0.05, 3.95),
    )
    assert not pool._pair_is_matched(
        plus, minus, position_tolerance=0.025, minimum_speed=0.08,
        target_qpos=np.asarray([0.8, 0.5]), history_qpos_bounds=(0.05, 3.95),
    )
    plus[5, 0] = -0.1
    assert not pool._pair_is_matched(
        plus, minus, position_tolerance=0.025, minimum_speed=0.08,
        target_qpos=np.asarray([0.4, 0.5]), history_qpos_bounds=(0.05, 3.95),
    )
