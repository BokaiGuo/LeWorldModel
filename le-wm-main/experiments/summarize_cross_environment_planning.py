"""Summarize the non-PushT planning evidence without pooling incompatible estimands.

The environments intentionally remain separate: TwoRoom and Reacher are live
RGB paired planning panels, Cube is a live paired failure-boundary intervention,
and PointMaze is an independent held-out offline planning gate.  This report is
an evidence index, not a universal success-rate aggregate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _live_panel(path: Path, environment: str) -> dict[str, Any]:
    payload = _read(path)
    expected = {
        "TwoRoom": "live_tworoom_planning_aggregate.v1",
        "Reacher": "live_reacher_planning_aggregate.v1",
    }[environment]
    if payload.get("schema") != expected:
        raise ValueError(f"{path}: expected {expected}, got {payload.get('schema')}")
    paired = payload["paired"]
    bootstrap = paired["bootstrap"]
    return {
        "environment": environment,
        "evidence_type": "live_rgb_paired_planning",
        "status": "positive" if bootstrap["ci95"][0] > 0 else "inconclusive",
        "n": int(bootstrap["n"]),
        "delta": float(paired["success_rate_delta"]),
        "ci95": [float(x) for x in bootstrap["ci95"]],
        "wins": int(paired["wins"]),
        "losses": int(paired["losses"]),
        "ties": int(paired["ties"]),
        "p_one_sided": paired.get("one_sided_exact_sign_test_p"),
        "artifact": str(path),
        "claim_boundary": payload.get("claim_boundary"),
    }


def _cube(path: Path) -> dict[str, Any]:
    payload = _read(path)
    if payload.get("schema") != "live_cube_phase_conditioned_planning.v1":
        raise ValueError(f"{path}: unexpected Cube schema")
    paired = payload["paired"]
    return {
        "environment": "Cube",
        "evidence_type": "live_rgb_paired_failure_boundary",
        "status": "negative_boundary",
        "n": len(paired["success_delta_phase_minus_baseline"]),
        "delta": float(paired["success_rate_delta"]),
        "ci95": None,
        "wins": int(paired["success_wins"]),
        "losses": int(paired["success_losses"]),
        "ties": int(paired["success_ties"]),
        "mean_final_distance_delta_m": float(paired["mean_final_distance_delta_m"]),
        "artifact": str(path),
        "claim_boundary": payload.get("claim_boundary"),
    }


def _pointmaze(path: Path) -> dict[str, Any]:
    payload = _read(path)
    if payload.get("schema") != "independent_planning_gain.v1":
        raise ValueError(f"{path}: unexpected PointMaze schema")
    inference = payload["paired_inference"]
    return {
        "environment": "PointMaze",
        "evidence_type": "heldout_offline_planning_gate",
        "status": "positive" if payload.get("positive_gain") and payload.get("statistical_gate") else "inconclusive",
        "n": int(inference["n_pairs"]),
        "delta": float(inference["mean"]),
        "ci95": [float(inference["ci95_low"]), float(inference["ci95_high"])],
        "p_one_sided": float(inference["p_one_sided"]),
        "protocol_gate": bool(payload.get("protocol_gate")),
        "statistical_gate": bool(payload.get("statistical_gate")),
        "artifact": str(path),
        "claim_boundary": payload.get("claim_boundary"),
    }


def _markdown(rows: list[dict[str, Any]], output: Path) -> str:
    lines = [
        "# Cross-environment planning evidence (non-PushT)",
        "",
        "The rows use their native estimands and are not pooled into one success rate.",
        "",
        "| Environment | Evidence | n | Delta | 95% CI | Status |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        ci = "—" if row.get("ci95") is None else "[%.4f, %.4f]" % tuple(row["ci95"])
        lines.append(
            f"| {row['environment']} | {row['evidence_type']} | {row['n']} | "
            f"{row['delta']:.4f} | {ci} | {row['status']} |"
        )
    lines += [
        "",
        "Interpretation: TwoRoom and Reacher provide positive live RGB paired "
        "consequences under their frozen protocols. Cube remains a strict "
        "negative boundary: prediction/action-interface interventions did not "
        "produce planning success. PointMaze remains statistically inconclusive "
        "under its independent held-out offline gate. These results do not "
        "support a universal world model or a universal transfer certificate.",
        "",
        f"Generated from `{output}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tworoom", type=Path, required=True)
    parser.add_argument("--reacher", type=Path, required=True)
    parser.add_argument("--cube", type=Path, required=True)
    parser.add_argument("--pointmaze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        _live_panel(args.tworoom, "TwoRoom"),
        _live_panel(args.reacher, "Reacher"),
        _cube(args.cube),
        _pointmaze(args.pointmaze),
    ]
    report = {
        "schema": "cross_environment_planning_evidence.v1",
        "environments": [row["environment"] for row in rows],
        "pooling": "forbidden_incompatible_estimands",
        "rows": rows,
        "claim_boundary": "cross-environment evidence index; not universal world-model or commutator certificate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(rows, args.output), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
