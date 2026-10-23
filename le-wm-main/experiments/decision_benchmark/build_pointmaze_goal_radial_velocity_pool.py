"""Materialize V15's fresh goal-centred, velocity-opposed PointMaze pool."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.decision_benchmark import build_pointmaze_velocity_matched_state_pool as pool


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "point_maze_v57_velocity_matched_history_20261023"
OUTPUT = ROOT / "data" / "point_maze_v58_goal_radial_velocity_history_20261023"
PAIR_COUNT, SEED = 416, 202611251
GOAL_QPOS, TARGET_RADIUS = (0.8, 0.8), 0.28


def main() -> int:
    report = pool.build_pool(
        INPUT, OUTPUT, pair_count=PAIR_COUNT, seed=SEED,
        target_goal=GOAL_QPOS, target_radius=TARGET_RADIUS,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
