"""Audit whether a sealed P0 model ordering agrees with physical execution.

This audit consumes immutable P0 and witness-replay receipts only.  It never
selects a candidate or changes a planner.  Its purpose is to distinguish a
replayed implementation error from a genuine frozen-model-to-physics boundary:
for every paired D2P/CEM root it compares the frozen terminal-cost ordering
with the evaluator-only final-qpos ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sign(value: float, *, atol: float) -> int:
    if not np.isfinite(value):
        raise ValueError("alignment values must be finite")
    return -1 if value < -atol else 1 if value > atol else 0


def audit(report: dict[str, Any], witness: dict[str, Any], *, atol: float = 1.0e-9) -> dict[str, Any]:
    """Return a read-only paired ordering audit for a P0 outcome receipt."""

    if report.get("schema") != "d2p_reacher_p0_goal_progress.v1" or not report.get("protocol", {}).get("outcome_only"):
        raise ValueError("alignment audit requires an outcome-only Reacher P0 receipt")
    if witness.get("schema") != "d2p_reacher_p0_witness_replay_audit.v1" or witness.get("status") != "pass":
        raise ValueError("alignment audit requires a passing witness-replay receipt")
    if not report.get("rows"):
        raise ValueError("P0 receipt has no roots")
    rows: list[dict[str, Any]] = []
    for row in report["rows"]:
        methods = row.get("methods", {})
        try:
            d2p = methods["d2p_goal_progress"]
            cem = methods["cem"]
            model_delta = float(d2p["predicted_terminal_cost"]) - float(cem["predicted_terminal_cost"])
            physical_delta = float(d2p["physical"]["final_qpos_distance"]) - float(cem["physical"]["final_qpos_distance"])
        except (KeyError, TypeError) as exc:
            raise ValueError("P0 root lacks D2P/CEM paired costs") from exc
        model_sign, physical_sign = _sign(model_delta, atol=atol), _sign(physical_delta, atol=atol)
        rows.append({
            "seed": int(row["seed"]),
            "model_d2p_minus_cem_terminal_cost": model_delta,
            "physical_d2p_minus_cem_final_qpos_distance": physical_delta,
            "model_order": "d2p" if model_sign < 0 else "cem" if model_sign > 0 else "tie",
            "physical_order": "d2p" if physical_sign < 0 else "cem" if physical_sign > 0 else "tie",
            "strict_order_agrees": bool(model_sign == physical_sign) if model_sign and physical_sign else None,
        })
    strict = [row for row in rows if row["strict_order_agrees"] is not None]
    agreement = int(sum(bool(row["strict_order_agrees"]) for row in strict))
    model_d2p = int(sum(row["model_order"] == "d2p" for row in rows))
    physical_d2p = int(sum(row["physical_order"] == "d2p" for row in rows))
    deltas = np.asarray([row["physical_d2p_minus_cem_final_qpos_distance"] for row in rows], dtype=np.float64)
    return {
        "schema": "d2p_reacher_p0_model_physical_alignment.v1",
        "status": "pass" if agreement == len(strict) else "model_physical_alignment_boundary",
        "scope": "post-outcome descriptive audit only; no qpos value may be fed back into candidate selection, damping choice, or a new P0/P1 protocol",
        "root_count": len(rows),
        "strict_ordered_root_count": len(strict),
        "strict_order_agreement_count": agreement,
        "strict_order_disagreement_count": len(strict) - agreement,
        "model_d2p_lower_cost_root_count": model_d2p,
        "physical_d2p_lower_distance_root_count": physical_d2p,
        "physical_d2p_minus_cem_mean": float(deltas.mean()),
        "leave_one_root_out_physical_means": [
            {"held_out_seed": row["seed"], "mean": float(np.delete(deltas, index).mean())}
            for index, row in enumerate(rows)
        ],
        "gates": {
            "all_witness_replay_gates_passed": bool(all(witness.get("gates", {}).values())),
            "all_paired_model_and_physical_values_finite": True,
            "witness_source_matches_report": bool(witness.get("source_report_sha256") == report.get("_sha256")),
        },
        "rows": rows,
        "claim_boundary": "A disagreement is a frozen-model-to-physical calibration boundary, not an action-replay defect or a license to tune against opened qpos outcomes.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--witness-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1.0e-9)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.atol <= 0:
        raise ValueError("atol must be positive")
    report_path, witness_path = args.report.resolve(), args.witness_audit.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["_sha256"] = _sha256(report_path)
    witness = json.loads(witness_path.read_text(encoding="utf-8"))
    result = audit(report, witness, atol=float(args.atol))
    result["source_report"] = str(report_path)
    result["source_report_sha256"] = report["_sha256"]
    result["witness_audit"] = str(witness_path)
    result["witness_audit_sha256"] = _sha256(witness_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "strict_order_agreement_count": result["strict_order_agreement_count"],
        "strict_order_disagreement_count": result["strict_order_disagreement_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
