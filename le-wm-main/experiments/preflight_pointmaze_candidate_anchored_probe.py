"""CPU-only RGB geometry preflight for the PointMaze V5 information action.

This is deliberately *not* a substitute for the CUDA closed-loop experiment.
It uses calibration RGB plus the environment initial state only to render a
declared initial observation and one causal no-op history block.  It then
checks whether V5 can form a task-safe, full-rank affine probe set from the
same fixed root candidate bank that the live RGB controller will rank.

The preflight answers a narrow but important protocol question before a costly
GPU run: is a future null result caused by an absent safe information action,
or by the learned repair/release path after such actions are available?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_live_pointmaze_planning import (
    DatasetMatchedRenderer,
    FRAME_STRIDE,
    GOAL_STATE,
    LegacyPointMazeRuntime,
    RGBMazeMap,
    action_bank,
    fit_visual_calibration,
    fit_visual_dynamics_calibration,
    green_center,
    load_episode,
    load_episode_states,
    planner_goal_from_rgb,
    task_safe_candidate_anchored_probe_actions,
)


SCHEMA = "pointmaze.candidate_anchored_probe_preflight.v1"


def _parse_ids(value: str, label: str) -> list[int]:
    values = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be a non-empty, duplicate-free comma-separated list")
    return values


def _record_root(
    *,
    root: int,
    state_root: Path,
    renderer: DatasetMatchedRenderer,
    calibration: Any,
    dynamics: Any,
    maze_map: RGBMazeMap,
    goal: np.ndarray,
    warmup_blocks: int,
    directions: int,
    magnitudes: list[float],
    probe_direction_caps: list[int],
    probe_magnitude_caps: list[float],
) -> dict[str, Any]:
    """Use environment state only for reset/stepping; all chart inputs are RGB."""

    runtime = LegacyPointMazeRuntime(load_episode_states(state_root, root)[0])
    image = renderer.render(runtime.state)
    center = green_center(image)
    previous_center = center
    for _ in range(warmup_blocks):
        runtime.step_block(np.zeros(2, dtype=np.float64))
        image = renderer.render(runtime.state)
        previous_center, center = center, green_center(image)
    position = calibration.position(center)
    velocity = calibration.velocity((center[0] - previous_center[0], center[1] - previous_center[1]))
    waypoints = maze_map.path_waypoints(position, goal)
    target = np.asarray(waypoints[0], dtype=np.float64) if waypoints else goal.copy()
    bank = action_bank(position, target, directions, magnitudes)
    configurations: list[dict[str, Any]] = []
    for count in probe_direction_caps:
        for magnitude in probe_magnitude_caps:
            actions, progress, indices = task_safe_candidate_anchored_probe_actions(
                position, target, velocity, dynamics, bank,
                probe_count=count, max_magnitude=magnitude,
            )
            rank = int(np.linalg.matrix_rank(np.c_[np.ones(len(actions)), actions])) if len(actions) else 0
            configurations.append({
                "probe_direction_cap": int(count),
                "probe_magnitude_cap": float(magnitude),
                "selected_probe_count": int(len(actions)),
                "affine_design_rank": rank,
                "candidate_indices": indices.astype(int).tolist(),
                "candidate_membership_exact": bool(np.array_equal(actions, bank[indices])),
                "predicted_goal_progress": progress.tolist(),
                "repairability_geometry_available": bool(
                    len(actions) >= 3 and rank == 3 and np.all(progress > 0.0)
                ),
            })
    return {
        "root": int(root),
        "position_from_rgb": position.tolist(),
        "velocity_from_rgb": velocity.tolist(),
        "target_from_rgb_map": target.tolist(),
        "fixed_candidate_bank_size": int(len(bank)),
        "configurations": configurations,
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("preflight requires at least one root record")
    keys = {
        (item["probe_direction_cap"], item["probe_magnitude_cap"])
        for record in records for item in record["configurations"]
    }
    summaries: list[dict[str, Any]] = []
    for count, magnitude in sorted(keys):
        items = [
            item for record in records for item in record["configurations"]
            if item["probe_direction_cap"] == count and item["probe_magnitude_cap"] == magnitude
        ]
        summaries.append({
            "probe_direction_cap": int(count),
            "probe_magnitude_cap": float(magnitude),
            "root_count": int(len(items)),
            "geometry_available_root_count": int(sum(item["repairability_geometry_available"] for item in items)),
            "geometry_available_rate": float(np.mean([
                item["repairability_geometry_available"] for item in items
            ])),
            "mean_selected_probe_count": float(np.mean([
                item["selected_probe_count"] for item in items
            ])),
            "exact_bank_membership_rate": float(np.mean([
                item["candidate_membership_exact"] for item in items
            ])),
        })
    return {"root_count": int(len(records)), "configurations": summaries}


def run(
    *,
    calibration_root: Path,
    state_root: Path,
    calibration_ids: list[int],
    roots: list[int],
    output: Path,
    warmup_blocks: int,
    directions: int,
    magnitudes: list[float],
    probe_direction_caps: list[int],
    probe_magnitude_caps: list[float],
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preflight artifact: {output}")
    images, _ = load_episode(calibration_root, calibration_ids[0])
    calibration = fit_visual_calibration(calibration_root, calibration_ids, max_frames=768)
    dynamics = fit_visual_dynamics_calibration(calibration_root, calibration_ids)
    renderer = DatasetMatchedRenderer.from_episode(images, calibration)
    maze_map = RGBMazeMap.from_renderer(renderer)
    goal_rgb = renderer.render(np.r_[GOAL_STATE, [0.0, 0.0]])
    goal = planner_goal_from_rgb(goal_rgb, calibration)
    records = [
        _record_root(
            root=root, state_root=state_root, renderer=renderer, calibration=calibration,
            dynamics=dynamics, maze_map=maze_map, goal=goal,
            warmup_blocks=warmup_blocks, directions=directions, magnitudes=magnitudes,
            probe_direction_caps=probe_direction_caps, probe_magnitude_caps=probe_magnitude_caps,
        )
        for root in roots
    ]
    report = {
        "schema": SCHEMA,
        "status": "completed_cpu_geometry_preflight",
        "claim_boundary": (
            "Calibration-RGB and fixed-bank geometry preflight only; not CUDA evidence, "
            "not a learned-policy release result, and not closed-loop task efficacy."
        ),
        "protocol": {
            "planner_input": "RGB-derived position, velocity, map waypoint, and calibration dynamics only",
            "environment_state_use": "initial reset and causal physics stepping only; never passed to probe selection",
            "warmup_blocks": int(warmup_blocks),
            "frame_stride": int(FRAME_STRIDE),
            "root_disjoint_from_calibration": not bool(set(calibration_ids) & set(roots)),
        },
        "calibration_roots": calibration_ids,
        "roots": roots,
        "summary": _summarize(records),
        "per_root": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--calibration-ids", required=True)
    parser.add_argument("--roots", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-blocks", type=int, default=1)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--magnitudes", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--probe-direction-caps", default="4,6")
    parser.add_argument("--probe-magnitude-caps", default="0.25,0.5")
    args = parser.parse_args()
    if args.warmup_blocks < 0 or args.directions < 4:
        raise ValueError("warmup blocks must be nonnegative and directions at least four")
    report = run(
        calibration_root=args.calibration_root,
        state_root=args.state_root,
        calibration_ids=_parse_ids(args.calibration_ids, "calibration IDs"),
        roots=_parse_ids(args.roots, "roots"), output=args.output,
        warmup_blocks=args.warmup_blocks, directions=args.directions,
        magnitudes=list(args.magnitudes),
        probe_direction_caps=_parse_ids(args.probe_direction_caps, "probe direction caps"),
        probe_magnitude_caps=[float(value) for value in args.probe_magnitude_caps.split(",")],
    )
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
