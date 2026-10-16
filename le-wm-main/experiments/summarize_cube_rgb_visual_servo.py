"""Aggregate independent paired RGB-only Cube visual-servo panels.

The input panels must be produced by the frozen protocol.  This script is an
analysis-only step: it never selects a controller from the test outcomes and
requires disjoint held-out seeds before pooling panels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_SCHEMAS = {"cube_rgb_visual_servo.v9", "cube_rgb_visual_servo.v10"}
PROTOCOL_FIELDS = (
    "environment",
    "approach_controller",
    "approach_threshold",
    "approach_pixel_threshold",
    "approach_standoff_m",
    "pixel_approach_gain",
    "world_action_gain",
    "push_controller",
    "push_pixel_gain",
    "probe_steps",
    "calibration_descent_steps",
    "runtime_input",
    "runtime_privileged_state",
    "evaluator_state_role",
    "camera_action_map_fit",
    "gripper_action_semantics",
    "root_disjoint",
)


def _sign_test(wins: int, losses: int) -> float | None:
    count = wins + losses
    if count == 0:
        return None
    lower_tail = sum(math.comb(count, k) for k in range(min(wins, losses) + 1)) / (2**count)
    return float(min(1.0, 2.0 * lower_tail))


def _bootstrap(values: np.ndarray, seed: int, draws: int) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(int(draws), values.size))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _summary(values: np.ndarray, seed: int, draws: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    wins = int(np.sum(values < 0.0))
    losses = int(np.sum(values > 0.0))
    ties = int(np.sum(values == 0.0))
    return {
        "n": int(values.size),
        "mean_delta_m": float(values.mean()) if values.size else None,
        "median_delta_m": float(np.median(values)) if values.size else None,
        "std_delta_m": float(values.std(ddof=1)) if values.size > 1 else None,
        "wins_improvement": wins,
        "losses_deterioration": losses,
        "ties": ties,
        "non_tie_improvement_rate": float(wins / (wins + losses)) if wins + losses else None,
        "bootstrap_ci95_mean_delta_m": _bootstrap(values, seed, draws),
        "two_sided_exact_sign_test_p": _sign_test(wins, losses),
    }


def _protocol_signature(protocol: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(protocol.get(field) for field in PROTOCOL_FIELDS)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unexpected schema in {path}: {payload.get('schema')!r}")
    protocol = payload["protocol"]
    paired = payload["paired"]
    seeds = np.asarray(paired["seed_order"], dtype=np.int64)
    if seeds.size == 0 or np.unique(seeds).size != seeds.size:
        raise ValueError(f"invalid or duplicated seed order in {path}")
    if int(protocol["test_episodes"]) != int(seeds.size):
        raise ValueError(f"test episode count mismatch in {path}")

    servo_rows = {int(row["seed"]): row for row in payload["visual_servo"]["episodes"]}
    baseline_rows = {int(row["seed"]): row for row in payload["no_lateral_baseline"]["episodes"]}
    if set(servo_rows) != set(seeds.tolist()) or set(baseline_rows) != set(seeds.tolist()):
        raise ValueError(f"paired seed mismatch in {path}")

    delta = np.asarray(
        [servo_rows[int(seed)]["final_goal_distance_m"] - baseline_rows[int(seed)]["final_goal_distance_m"] for seed in seeds],
        dtype=np.float64,
    )
    declared_delta = np.asarray(paired["final_distance_delta_servo_minus_baseline_m"], dtype=np.float64)
    if not np.allclose(delta, declared_delta, atol=1e-10, rtol=1e-10):
        raise ValueError(f"declared paired deltas do not reproduce from episode rows in {path}")
    task_ids = np.asarray([servo_rows[int(seed)]["task_id"] for seed in seeds], dtype=np.int64)
    return {
        "path": str(path),
        "input_schema": payload["schema"],
        "protocol": protocol,
        "seeds": seeds,
        "delta": delta,
        "task_ids": task_ids,
        "calibration_roots": int(protocol["calibration_roots"]),
        "test_episodes": int(protocol["test_episodes"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20262220)
    args = parser.parse_args()
    if args.bootstrap_draws < 1000:
        raise ValueError("use at least 1000 bootstrap draws")

    panels = [_load(path) for path in args.input]
    signature = _protocol_signature(panels[0]["protocol"])
    for panel in panels[1:]:
        if _protocol_signature(panel["protocol"]) != signature:
            raise ValueError(f"frozen protocol mismatch: {panel['path']}")

    seeds = np.concatenate([panel["seeds"] for panel in panels])
    if np.unique(seeds).size != seeds.size:
        raise ValueError("held-out seeds overlap across input panels")
    deltas = np.concatenate([panel["delta"] for panel in panels])
    task_ids = np.concatenate([panel["task_ids"] for panel in panels])

    panel_reports = []
    for index, panel in enumerate(panels):
        panel_reports.append({
            "artifact": panel["path"],
            "test_seeds": panel["seeds"].tolist(),
            "calibration_roots": panel["calibration_roots"],
            "test_episodes": panel["test_episodes"],
            "paired_servo_minus_baseline": _summary(
                panel["delta"], args.bootstrap_seed + index, args.bootstrap_draws,
            ),
        })

    per_task: dict[str, Any] = {}
    for task_id in sorted(np.unique(task_ids).tolist()):
        mask = task_ids == int(task_id)
        per_task[str(int(task_id))] = {
            "episodes": int(mask.sum()),
            "paired_servo_minus_baseline": _summary(
                deltas[mask], args.bootstrap_seed + 100 + int(task_id), args.bootstrap_draws,
            ),
        }

    report = {
        "schema": "cube_rgb_visual_servo.aggregate.v1",
        "protocol": {
            "input_artifacts": [panel["path"] for panel in panels],
            "input_schemas": [panel["input_schema"] for panel in panels],
            "independent_test_seed_panels": True,
            "pooled_test_episodes": int(seeds.size),
            "pooled_seed_order": seeds.tolist(),
            "bootstrap_draws": int(args.bootstrap_draws),
            "bootstrap_seed": int(args.bootstrap_seed),
            "planning_outcome_used_for_selection": False,
            "protocol_fields_checked": list(PROTOCOL_FIELDS),
        },
        "frozen_protocol": {field: panels[0]["protocol"].get(field) for field in PROTOCOL_FIELDS},
        "panels": panel_reports,
        "pooled": {
            "paired_servo_minus_baseline": _summary(deltas, args.bootstrap_seed, args.bootstrap_draws),
            "calibration_roots_total": int(sum(panel["calibration_roots"] for panel in panels)),
            "test_episodes_total": int(seeds.size),
        },
        "per_task": per_task,
        "claim_boundary": (
            "Pooled paired RGB-only Cube visual-servo consequence for a frozen "
            "world-approach, threshold-0.015, pixel-adaptive-push protocol. "
            "This is an environment/task-scoped contact-closure result; it does "
            "not establish frozen LeWM visual planning, universal transfer, "
            "causal latent dynamics, stable pose closure, or a general certificate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pooled": report["pooled"], "per_task": report["per_task"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
