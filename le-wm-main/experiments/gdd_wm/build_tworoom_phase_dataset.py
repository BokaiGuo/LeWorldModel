"""Attach outcome-free RGB phase labels to a TwoRoom candidate panel."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import gymnasium as gym

from experiments.run_live_tworoom_planning import (
    TwoRoomGoalAliasWrapper,
    _rgb_red_centroid,
    _rgb_reactive_action,
)
from experiments.audit_tworoom_rgb_reactive_controller import _door_point


def build(input_path: Path, output_path: Path, *, render_seed: int | None = None) -> dict:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"{input_path}: expected candidate dataset with records")
    records = payload["records"]
    roots: dict[str, int] = {}
    for row in records:
        root = str(row["instance_id"])
        seed = int(row.get("start_episode_id", root.rsplit("_", 1)[-1]))
        roots[root] = seed if render_seed is None else int(render_seed) + len(roots)
    env = TwoRoomGoalAliasWrapper(gym.make("swm/TwoRoom-v1", render_mode="rgb_array"))
    phases: dict[str, str] = {}
    waypoint_directions: dict[str, list[float]] = {}
    try:
        for root, seed in roots.items():
            _, info = env.reset(seed=seed)
            frame = env.render()
            phase_action, phase = _rgb_reactive_action(frame, info["goal"])
            if phase not in {"door", "goal"}:
                raise ValueError(f"unexpected TwoRoom phase {phase!r}")
            phases[root] = phase
            source = _rgb_red_centroid(frame)
            goal = _rgb_red_centroid(info["goal"])
            door, _ = _door_point(frame)
            waypoint = door if phase == "door" else goal
            direction = waypoint - source
            norm = float((direction ** 2).sum() ** 0.5)
            if norm <= 1.0e-8:
                raise ValueError(f"{root}: RGB waypoint direction is degenerate")
            waypoint_directions[root] = (direction / norm).astype(float).tolist()
    finally:
        env.close()
    output = copy.deepcopy(payload)
    for row in output["records"]:
        row["task_phase"] = phases[str(row["instance_id"])]
        row["task_waypoint_direction"] = waypoint_directions[str(row["instance_id"])]
    metadata = dict(output.get("metadata", {}))
    metadata.update({
        "phase_conditioning": "rgb_reactive_door_goal_v1",
        "phase_label_source": "current_rgb_and_rendered_goal_only",
        "phase_label_outcome_free": True,
        "phase_root_count": len(phases),
        "waypoint_conditioning": "rgb_reactive_phase_waypoint_direction_v1",
        "waypoint_direction_source": "current_rgb_and_rendered_goal_only",
        "waypoint_direction_outcome_free": True,
    })
    output["metadata"] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"records": len(records), "roots": len(phases), "phase_counts": {
        "door": sum(value == "door" for value in phases.values()),
        "goal": sum(value == "goal" for value in phases.values()),
    }}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
