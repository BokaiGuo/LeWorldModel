"""Audit whether a passive local release actually changes a PointMaze action.

``local_verified_release`` means that the residual chart passed its local
validity conditions.  It is not, by itself, evidence that the chart supplied
new decision information: its selected candidate can be identical to the
visual fallback.  This post-hoc audit keeps those two facts separate without
using any task outcome to select an action or tune a threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "pointmaze.passive_local.mechanism_audit.v1"


def _rows(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = payload.get("rows", [])
    result = {int(row["episode_id"]): row for row in rows}
    if not result or len(result) != len(rows):
        raise ValueError("expected non-empty rows with unique episode IDs")
    return result


def _canonical_action(trace: dict[str, Any]) -> tuple[float, ...] | None:
    action = trace.get("selected_action")
    if action is None:
        return None
    try:
        return tuple(float(value) for value in action)
    except (TypeError, ValueError) as error:
        raise ValueError("selected_action must be a numeric sequence") from error


def audit(passive: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Return a root-paired action-divergence audit without outcome tuning."""

    passive_rows, baseline_rows = _rows(passive), _rows(baseline)
    if set(passive_rows) != set(baseline_rows):
        raise ValueError("passive and baseline reports must share exact roots")
    nominal_releases = 0
    same_candidate_releases = 0
    divergent_candidate_releases = 0
    support_fractions: list[float] = []
    roots_with_any_action_difference = 0
    action_difference_blocks = 0
    roots_with_nominal_release = 0
    for root in sorted(passive_rows):
        passive_trace = passive_rows[root].get("trace", [])
        baseline_trace = baseline_rows[root].get("trace", [])
        if len(passive_trace) != len(baseline_trace):
            raise ValueError(f"root {root}: trace lengths differ")
        root_nominal_release = False
        root_action_difference = False
        for block, (local, visual) in enumerate(zip(passive_trace, baseline_trace)):
            gate = local.get("action_transfer_gate", {})
            if bool(gate.get("local_verified_release", False)):
                nominal_releases += 1
                root_nominal_release = True
                local_index = gate.get("local_affine_selected_index")
                visual_index = gate.get("local_affine_visual_index")
                selected_index = local.get("selected_index")
                if local_index is None or visual_index is None or selected_index is None:
                    raise ValueError(f"root {root} block {block}: incomplete local release trace")
                if int(selected_index) != int(local_index):
                    raise ValueError(f"root {root} block {block}: release did not execute local candidate")
                support_fractions.append(float(gate["local_affine_supported_candidate_fraction"]))
                if int(local_index) == int(visual_index):
                    same_candidate_releases += 1
                else:
                    divergent_candidate_releases += 1
            if _canonical_action(local) != _canonical_action(visual):
                root_action_difference = True
                action_difference_blocks += 1
        roots_with_nominal_release += int(root_nominal_release)
        roots_with_any_action_difference += int(root_action_difference)
    if nominal_releases != same_candidate_releases + divergent_candidate_releases:
        raise RuntimeError("release accounting does not partition nominal releases")
    return {
        "schema": SCHEMA,
        "root_count": len(passive_rows),
        "nominal_local_release_count": nominal_releases,
        "same_candidate_release_count": same_candidate_releases,
        "divergent_candidate_release_count": divergent_candidate_releases,
        "roots_with_nominal_local_release": roots_with_nominal_release,
        "roots_with_any_executed_action_difference_vs_htsc": roots_with_any_action_difference,
        "executed_action_difference_block_count_vs_htsc": action_difference_blocks,
        "mean_supported_candidate_fraction_at_release": (
            None if not support_fractions else sum(support_fractions) / len(support_fractions)
        ),
        "interpretation": (
            "A nominal local release is not an effective decision intervention unless its candidate "
            "differs from the visual fallback.  This audit is descriptive only and never tunes a policy."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passive-report", type=Path, required=True)
    parser.add_argument("--htsc-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit artifact: {args.output}")
    passive = json.loads(args.passive_report.read_text(encoding="utf-8"))
    htsc = json.loads(args.htsc_report.read_text(encoding="utf-8"))
    report = audit(passive, htsc)
    report.update({
        "passive_report": str(args.passive_report.resolve()),
        "htsc_report": str(args.htsc_report.resolve()),
        "outcome_data_used_for_policy_selection": False,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
