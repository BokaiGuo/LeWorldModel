"""Run the privileged OGBench Cube Markov-oracle ceiling on frozen roots.

This is an evaluator-only ceiling.  The oracle reads simulator state and is
never a deployable RGB policy; it exists to separate task/action reachability
from the RGB-only carry-transfer failure.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_live_cube_rgb_pick_place import _numpyify


def _run(initial: dict[str, Any], max_steps: int) -> dict[str, Any]:
    env = gym.make(
        "swm/OGBCube-v0", ob_type="pixels", render_mode="rgb_array",
        terminate_at_goal=False,
    )
    try:
        obs, info = env.reset(
            seed=int(initial["seed"]),
            options={
                "task_id": int(initial["task_id"]),
                "render_goal": True,
                "variation_values": _numpyify(copy.deepcopy(initial["variation_values"])),
                "state": np.asarray(initial["state"], dtype=np.float64),
            },
        )
        goal = np.asarray(env.unwrapped.cur_task_info["goal_xyzs"][0], dtype=np.float64)
        oracle = CubeMarkovOracle(env=env.unwrapped, min_norm=0.4, max_step=max_steps)
        oracle.reset(obs, info)
        terminated = truncated = False
        for step in range(int(max_steps)):
            # The current stable-worldmodel wrapper omits these oracle aliases;
            # reconstructing them from the evaluator info is still privileged.
            info["privileged/target_block"] = 0
            info["privileged/target_block_pos"] = goal
            info["privileged/target_block_yaw"] = info["privileged/block_0_yaw"]
            action = oracle.select_action(obs, info)
            obs, _reward, terminated, truncated, info = env.step(
                np.asarray(action, dtype=np.float32)
            )
            if terminated or truncated:
                break
        block = np.asarray(info["privileged/block_0_pos"], dtype=np.float64)
        distance = float(np.linalg.norm(block[:2] - goal[:2]))
        return {
            "seed": int(initial["seed"]),
            "task_id": int(initial["task_id"]),
            "steps": int(step + 1),
            "distance_m": distance,
            "success": bool(distance <= 0.05),
            "block_z": float(block[2]),
            "contact": float(np.max(np.asarray(info["proprio/gripper_contact"]))),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-conditions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()
    source = json.loads(args.initial_conditions.read_text())
    initials = source["initial_conditions"]
    rows = [_run(initial, args.max_steps) for initial in initials]
    payload = {
        "schema": "cube_markov_oracle_ceiling.v1",
        "claim_boundary": "privileged simulator-state ceiling; not RGB or learned-world-model evidence",
        "protocol": {
            "initial_conditions": str(args.initial_conditions),
            "root_disjoint": True,
            "max_steps": int(args.max_steps),
            "goal_tolerance_m": 0.05,
            "oracle": "ogbench CubeMarkovOracle",
        },
        "rows": rows,
        "aggregate": {
            "roots": len(rows),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_distance_m": float(np.mean([row["distance_m"] for row in rows])),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
