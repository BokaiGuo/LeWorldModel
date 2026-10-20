"""Build a train-only, phase-specific waypoint-cost view of a TwoRoom panel.

The panel structure and RGB/model fields are preserved.  Only the evaluator
label is replaced by a predeclared task utility: during the door phase the
cost is RGB marker distance to the visible door at the midpoint of the pulse;
during the goal phase the original terminal task cost is retained.  The
simulator is used only to create offline labels and never at deployment.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.audit_tworoom_rgb_reactive_controller import _door_point
from experiments.run_live_tworoom_planning import (
    TwoRoomGoalAliasWrapper,
    _rgb_red_centroid,
    _rgb_reactive_action,
)


def build(input_path: Path, output_path: Path) -> dict:
    # TwoRoom's renderer performs small tensor compositing operations.  The
    # default OpenMP thread pool makes thousands of these tiny calls dominate
    # this offline label pass; keep the simulator deterministic and avoid
    # thread-pool oversubscription.  This is label construction only, not a
    # replacement for the CUDA teacher fit.
    torch.set_num_threads(1)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"{input_path}: expected a candidate dataset with records")
    rows = payload["records"]
    by_root: dict[str, list[dict]] = {}
    for row in rows:
        by_root.setdefault(str(row["instance_id"]), []).append(row)

    env = TwoRoomGoalAliasWrapper(gym.make("swm/TwoRoom-v1", render_mode="rgb_array"))
    labels: dict[tuple[str, str], float] = {}
    phase_counts = {"door": 0, "goal": 0}
    try:
        for root, root_rows in by_root.items():
            seed = int(root.rsplit("_", 1)[-1])
            _, info = env.reset(seed=seed)
            initial = np.asarray(env.render(), dtype=np.uint8)
            _, phase = _rgb_reactive_action(initial, info["goal"])
            phase_counts[phase] += 1
            door, _ = _door_point(initial)
            goal = _rgb_red_centroid(np.asarray(info["goal"], dtype=np.uint8))
            midpoint = max(1, len(root_rows[0]["action_sequence"]) // 2)
            for row in root_rows:
                env.reset(seed=seed)
                observed_midpoint = None
                terminated = truncated = False
                for step, action in enumerate(np.asarray(row["action_sequence"], dtype=np.float32), start=1):
                    _, _, terminated, truncated, _ = env.step(action)
                    if step == midpoint:
                        observed_midpoint = _rgb_red_centroid(np.asarray(env.render(), dtype=np.uint8))
                    if terminated or truncated:
                        break
                final = _rgb_red_centroid(np.asarray(env.render(), dtype=np.uint8))
                if phase == "door":
                    if observed_midpoint is None:
                        observed_midpoint = final
                    cost = float(np.linalg.norm(observed_midpoint - door))
                else:
                    cost = float(np.linalg.norm(final - goal))
                labels[(root, str(row["candidate_id"]))] = cost
    finally:
        env.close()

    output = copy.deepcopy(payload)
    for row in output["records"]:
        key = (str(row["instance_id"]), str(row["candidate_id"]))
        row["real_task_cost"] = labels[key]
        row["waypoint_target_kind"] = "midpoint_door_distance" if row.get("task_phase") == "door" else "terminal_goal_distance"
    metadata = dict(output.get("metadata", {}))
    metadata.update({
        "waypoint_target": "door_midpoint_else_terminal_goal_v1",
        "waypoint_target_outcome_free_at_deployment": True,
        "waypoint_target_label_role": "offline_evaluator_only",
        "waypoint_target_phase_counts": phase_counts,
        "source_task_cost_preserved": False,
    })
    output["metadata"] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"records": len(rows), "roots": len(by_root), "phase_counts": phase_counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
