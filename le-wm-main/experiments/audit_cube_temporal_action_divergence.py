"""Audit whether temporal and static Cube rankers actually change actions.

This diagnostic separates representation use from downstream transfer: a
different selected action is not counted as a task gain unless the final
paired block outcome changes as well.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "cube_temporal_closed_loop.v1":
        raise ValueError(f"unexpected schema in {path}")
    static = {int(row["seed"]): row for row in payload["static"]["episodes"]}
    temporal = {int(row["seed"]): row for row in payload["temporal"]["episodes"]}
    if set(static) != set(temporal):
        raise ValueError(f"static/temporal seed mismatch in {path}")
    rows = []
    for seed in payload["paired"]["seed_order"]:
        seed = int(seed)
        static_actions = np.asarray(static[seed]["actions"], dtype=np.float64)
        temporal_actions = np.asarray(temporal[seed]["actions"], dtype=np.float64)
        if static_actions.shape != temporal_actions.shape:
            raise ValueError(f"action trace shape mismatch for seed {seed} in {path}")
        changed = np.linalg.norm(static_actions - temporal_actions, axis=1) > 1.0e-8
        delta = float(
            temporal[seed]["final_goal_distance_m"]
            - static[seed]["final_goal_distance_m"]
        )
        rows.append({
            "seed": seed,
            "task_id": int(static[seed]["task_id"]),
            "steps": int(len(changed)),
            "changed_steps": int(changed.sum()),
            "changed_fraction": float(changed.mean()),
            "final_distance_delta_m": delta,
            "outcome_changed": bool(delta != 0.0),
        })
    return {"path": str(path), "rows": rows}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    changed = np.asarray([row["changed_steps"] for row in rows], dtype=np.float64)
    fractions = np.asarray([row["changed_fraction"] for row in rows], dtype=np.float64)
    deltas = np.asarray([row["final_distance_delta_m"] for row in rows], dtype=np.float64)
    changed_mask = changed > 0
    return {
        "episodes": int(len(rows)),
        "episodes_with_action_change": int(changed_mask.sum()),
        "action_change_episode_rate": float(changed_mask.mean()) if len(rows) else None,
        "mean_changed_steps": float(changed.mean()) if len(rows) else None,
        "mean_changed_fraction": float(fractions.mean()) if len(rows) else None,
        "episodes_with_outcome_change": int(np.sum(deltas != 0.0)),
        "mean_final_distance_delta_m": float(deltas.mean()) if len(rows) else None,
        "mean_delta_when_actions_changed_m": float(deltas[changed_mask].mean()) if changed_mask.any() else None,
        "mean_delta_when_actions_equal_m": float(deltas[~changed_mask].mean()) if (~changed_mask).any() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panels = [_load(path) for path in args.input]
    rows = [row for panel in panels for row in panel["rows"]]
    if len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("input panels contain overlapping test seeds")
    by_task: dict[str, Any] = {}
    for task_id in sorted({row["task_id"] for row in rows}):
        by_task[str(task_id)] = _summarize([row for row in rows if row["task_id"] == task_id])
    report = {
        "schema": "cube_temporal_action_divergence.v1",
        "protocol": {
            "input_artifacts": [panel["path"] for panel in panels],
            "test_seed_panels_disjoint": True,
            "action_change_tolerance_l2": 1.0e-8,
            "evaluator_state_used": False,
        },
        "pooled": _summarize(rows),
        "per_task": by_task,
        "episodes": rows,
        "claim_boundary": (
            "Diagnostic separation of action-selection change from downstream "
            "Cube transfer; it does not establish causal latent dynamics or "
            "universal visual planning."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pooled": report["pooled"], "per_task": report["per_task"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
