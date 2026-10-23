"""Materialize a fresh V16 radial velocity pool with causal RGB visibility."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.decision_benchmark import build_pointmaze_velocity_matched_state_pool as pool


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "point_maze_v58_goal_radial_velocity_history_20261023"
OUTPUT = ROOT / "data" / "point_maze_v59_goal_radial_visible_velocity_history_20261023"
PAIR_COUNT, SEED = 416, 202611261
GOAL_QPOS, TARGET_RADIUS, VELOCITY_MAGNITUDE = (0.8, 0.8), 0.20, 0.75
HISTORY_QPOS_BOUNDS = (0.05, 3.95)


def main() -> int:
    report = pool.build_pool(
        INPUT, OUTPUT, pair_count=PAIR_COUNT, seed=SEED,
        target_goal=GOAL_QPOS, target_radius=TARGET_RADIUS,
        velocity_magnitude=VELOCITY_MAGNITUDE, history_qpos_bounds=HISTORY_QPOS_BOUNDS,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
