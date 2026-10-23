"""Build a fresh high-velocity radial pool under the causal visibility contract."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.decision_benchmark import build_pointmaze_velocity_matched_state_pool as pool


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "point_maze_v59_goal_radial_visible_velocity_history_20261023"
OUTPUT = ROOT / "data" / "point_maze_v60_goal_radial_visible_high_velocity_history_20261023"


def main() -> int:
    report = pool.build_pool(
        INPUT, OUTPUT, pair_count=416, seed=202611271,
        target_goal=(0.8, 0.8), target_radius=0.30, velocity_magnitude=1.5,
        history_qpos_bounds=(0.05, 3.95),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
