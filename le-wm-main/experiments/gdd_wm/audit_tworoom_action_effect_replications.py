"""Reject falsely independent TwoRoom action-effect replication panels.

An output directory name or an additional report is not an independent
replication.  This audit requires pairwise-disjoint held-out roots and records
the content hash of the associated intervention panel when it is available.
It is intentionally read-only: it cannot turn a reused panel into evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any


SCHEMA = "tworoom_action_effect_replication_audit.v1"
REPORT_SCHEMA = "tworoom_action_effect_repairability.v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"{path}: unexpected report schema")
    outcome = value.get("protocol", {}).get("root_ids", {}).get("outcome")
    if not isinstance(outcome, list) or not outcome:
        raise ValueError(f"{path}: missing nonempty outcome root IDs")
    return value


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(report_paths: list[Path]) -> dict[str, Any]:
    if len(report_paths) < 2:
        raise ValueError("need at least two reports to audit replication independence")
    rows: list[dict[str, Any]] = []
    for path in report_paths:
        report = _load(path)
        roots = tuple(sorted({int(root) for root in report["protocol"]["root_ids"]["outcome"]}))
        if len(roots) != len(report["protocol"]["root_ids"]["outcome"]):
            raise ValueError(f"{path}: outcome roots contain duplicates")
        rows.append({
            "report": str(path.resolve()),
            "outcome_roots": list(roots),
            "outcome_count": len(roots),
            "panel_arrays_sha256": _sha256(path.parent / "panel_arrays.npz"),
        })
    pairs: list[dict[str, Any]] = []
    for left, right in combinations(rows, 2):
        shared = sorted(set(left["outcome_roots"]) & set(right["outcome_roots"]))
        pairs.append({
            "left": left["report"],
            "right": right["report"],
            "shared_outcome_roots": shared,
            "shared_outcome_count": len(shared),
            "same_outcome_panel": bool(shared) and len(shared) == left["outcome_count"] == right["outcome_count"],
            "same_panel_arrays_sha256": (
                left["panel_arrays_sha256"] is not None
                and left["panel_arrays_sha256"] == right["panel_arrays_sha256"]
            ),
        })
    independent = all(pair["shared_outcome_count"] == 0 for pair in pairs)
    return {
        "schema": SCHEMA,
        "status": "passed_pairwise_independent" if independent else "rejected_nonindependent_outcome_panels",
        "pairwise_independent": independent,
        "reports": rows,
        "pairs": pairs,
        "claim_boundary": (
            "This audit establishes only whether declared outcome panels are independent. "
            "It does not validate any policy, model, effect adapter, or planning gain."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite replication audit: {args.output}")
    result = audit(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "pairwise_independent": result["pairwise_independent"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
