"""Read-only audit: was P1 less goal-greedy than its physical zero-prefix?"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit(report: dict[str, Any], *, tolerance: float = 1.0e-8) -> dict[str, Any]:
    if report.get("schema") != "d2p_reacher_p1_reconfiguration.v1":
        raise ValueError("input is not a Reacher D2P P1 receipt")
    rows: list[dict[str, Any]] = []
    for row in report.get("rows", []):
        method = row["methods"]["reconfiguration"]
        receipt, physical = method["first_receipt"], method["physical"]
        eligible = bool(receipt["non_greedy_witness_found"])
        zero_after = float(physical["after_zero_prefix_qpos_distance"])
        after = float(physical["after_first_chunk_qpos_distance"])
        rows.append({"seed": int(row["seed"]), "model_eligible": eligible, "candidate_minus_zero_prefix_distance": after - zero_after, "physically_non_greedy_relative_to_zero_prefix": bool(after >= zero_after - tolerance)})
    eligible = [row for row in rows if row["model_eligible"]]
    passing = [row for row in eligible if row["physically_non_greedy_relative_to_zero_prefix"]]
    gates = {"all_model_eligible_roots_physically_non_greedy_relative_to_zero_prefix": len(passing) == len(eligible), "at_least_one_model_eligible_root": bool(eligible)}
    return {"schema": "d2p_reacher_p1_physical_nongreediness_audit.v2", "status": "pass" if all(gates.values()) else "fail", "source_schema": report["schema"], "tolerance": tolerance, "gates": gates, "eligible_root_count": len(eligible), "physically_non_greedy_eligible_root_count": len(passing), "rows": rows, "claim_boundary": "A failing audit invalidates a reconfiguration mechanism claim for the source receipt; it does not establish a world-model theory negative."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {args.output}")
    result = audit(json.loads(args.report.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
