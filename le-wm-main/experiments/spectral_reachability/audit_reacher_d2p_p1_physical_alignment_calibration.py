"""Reduce calibration-only P1 receipts without turning a gate failure into theory evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PHYSICAL_GATE = "every_model_eligible_reconfiguration_is_physically_non_greedy_relative_to_zero_prefix"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(reports: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    """Return a fail-closed calibration reducer for predeclared model-only bands."""

    if not reports:
        raise ValueError("physical-alignment calibration audit requires at least one receipt")
    rows: list[dict[str, Any]] = []
    for path, report in reports:
        if report.get("schema") != "d2p_reacher_p1_reconfiguration.v1":
            raise ValueError(f"{path}: not a Reacher D2P P1 receipt")
        protocol, gates = report.get("protocol", {}), report.get("gates", {})
        if report.get("status") != "calibration_only" or not protocol.get("calibration_only"):
            raise ValueError(f"{path}: physical-alignment audit accepts calibration-only receipts")
        if PHYSICAL_GATE not in gates:
            raise ValueError(f"{path}: missing physical non-greediness gate")
        nonphysical_gates = {key: value for key, value in gates.items() if key != PHYSICAL_GATE}
        eligibility = report.get("eligibility", {})
        rows.append({
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "non_greedy_margin": protocol.get("non_greedy_margin"),
            "max_non_greedy_margin": protocol.get("max_non_greedy_margin"),
            "max_relative_reachability_improvement": protocol.get("max_relative_reachability_improvement"),
            "min_second_to_first_singular_ratio": protocol.get("min_second_to_first_singular_ratio"),
            "eligible_root_count": eligibility.get("eligible_root_count"),
            "physically_non_greedy_eligible_root_count": eligibility.get("physically_non_greedy_eligible_root_count"),
            "all_nonphysical_gates_pass": all(nonphysical_gates.values()),
            "physical_nongreediness_gate_pass": bool(gates[PHYSICAL_GATE]),
            "passes_calibration": all(gates.values()),
        })
    any_pass = any(bool(row["passes_calibration"]) for row in rows)
    return {
        "schema": "d2p_reacher_p1_physical_alignment_calibration_audit.v1",
        "status": "candidate_found" if any_pass else "blocked_model_to_physical_alignment",
        "candidate_count": len(rows),
        "any_model_only_band_passes_physical_calibration": any_pass,
        "candidates": rows,
        "claim_boundary": (
            "A blocked result prevents reopening the P1 efficacy panel for these frozen model-only "
            "admissibility bands. It is a model-to-physical short-horizon calibration boundary, not a "
            "world-model theory negative or a comparison of final P1 efficacy."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable audit output: {args.output}")
    reports = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in args.report]
    result = audit(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
