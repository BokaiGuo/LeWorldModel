"""Summarize a paired Cube chart/effect/hybrid intervention.

The summary is root-paired and descriptive. It never selects a controller from
the outcome roots and does not turn a positive mechanism result into frozen-
LeWM or universal planning evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _rows(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    return {int(row["seed"]): row for row in rows if "seed" in row}


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least baseline and one intervention are required")
    by_name = {path.stem: _rows(path) for path in paths}
    baseline_name = paths[0].stem
    baseline = by_name[baseline_name]
    if not baseline:
        raise ValueError("baseline contains no root rows")

    arms: dict[str, Any] = {}
    for name, rows in by_name.items():
        shared = sorted(set(baseline) & set(rows))
        success = [bool(rows[root].get("success", False)) for root in shared]
        contact = [bool(rows[root].get("contact_observed", False)) for root in shared]
        distances = [
            float(rows[root]["final"]["block_goal_distance_m"])
            for root in shared
            if rows[root].get("final", {}).get("block_goal_distance_m") is not None
        ]
        effect_ranks = [
            int(rows[root].get("effect_rank", 0))
            for root in shared
            if rows[root].get("effect_rank") is not None
        ]
        arms[name] = {
            "paired_roots": len(shared),
            "success_rate": float(np.mean(success)) if success else None,
            "contact_rate": float(np.mean(contact)) if contact else None,
            "mean_final_distance_m": _mean(distances),
            "mean_effect_rank": _mean([float(v) for v in effect_ranks]),
            "roots": shared,
        }

    paired: dict[str, Any] = {}
    for name, rows in by_name.items():
        if name == baseline_name:
            continue
        shared = sorted(set(baseline) & set(rows))
        distance_delta = []
        success_delta = []
        contact_delta = []
        for root in shared:
            base_distance = baseline[root].get("final", {}).get("block_goal_distance_m")
            arm_distance = rows[root].get("final", {}).get("block_goal_distance_m")
            if base_distance is not None and arm_distance is not None:
                distance_delta.append(float(base_distance) - float(arm_distance))
            success_delta.append(
                int(bool(rows[root].get("success", False)))
                - int(bool(baseline[root].get("success", False)))
            )
            contact_delta.append(
                int(bool(rows[root].get("contact_observed", False)))
                - int(bool(baseline[root].get("contact_observed", False)))
            )
        paired[name] = {
            "paired_roots": len(shared),
            "distance_improvement_m_positive_is_better": _mean(distance_delta),
            "success_rate_delta": _mean([float(v) for v in success_delta]),
            "contact_rate_delta": _mean([float(v) for v in contact_delta]),
            "per_root_distance_improvement_m": distance_delta,
        }
    return {
        "schema": "cube_action_effect_panel_summary.v1",
        "baseline": baseline_name,
        "arms": arms,
        "paired_deltas": paired,
        "claim_boundary": (
            "root-paired RGB mechanism intervention; descriptive action-effect "
            "diagnostic, not frozen-LeWM or universal planning evidence"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
