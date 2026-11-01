"""Exact root-ID comparison of strict-render partials and a new RGB audit.

Only the overlapping, completed reference roots are covered. This cannot
retroactively turn a failed strict-render run into a completed confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare_roots(reference: list[dict], candidate: list[dict]) -> dict:
    def index(rows):
        result = {int(row["seed"]): row for row in rows}
        if len(result) != len(rows) or not rows:
            raise ValueError("nonempty duplicate-free root IDs required")
        return result
    left, right = index(reference), index(candidate)
    if not left.keys() <= right.keys():
        raise ValueError("candidate report lacks completed reference roots")
    fields = ("source_episode", "source_start_row", "root_warmup_policy", "root_rgb_sha256",
              "goal_rgb_sha256", "proposal_support", "candidate_raw_packets", "successor_rgb_sha256")
    rows = []
    for seed in sorted(left):
        a, b = left[seed], right[seed]
        checks = {key: a[key] == b[key] for key in fields}
        policies_a = {item["path"]: item for item in a["policies"]}
        policies_b = {item["path"]: item for item in b["policies"]}
        if len(policies_a) != len(a["policies"]) or len(policies_b) != len(b["policies"]) or policies_a.keys() != policies_b.keys():
            raise ValueError("unique, identical policy identities required")
        for path in policies_a:
            for key in ("predicted_rgb_goal_costs", "observed_rgb_goal_costs"):
                x, y = np.asarray(policies_a[path][key]), np.asarray(policies_b[path][key])
                checks[path + ":" + key] = bool(np.isfinite(x).all() and np.isfinite(y).all() and np.array_equal(x, y))
        rows.append({"seed": seed, "exact": all(checks.values()), "checks": checks})
    return {"status": "pass" if all(row["exact"] for row in rows) else "fail",
            "matched_root_count": len(rows), "rows": rows,
            "claim_boundary": "Exact equivalence on completed reference roots only; failed strict-render roots remain execution failures, not ranking results or efficacy."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-roots", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--expected-reference-roots", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.expected_reference_roots < 1:
        raise ValueError("positive root count and new output required")
    files = sorted(args.reference_roots.glob("root_*.json"))
    if len(files) != args.expected_reference_roots:
        raise ValueError("unexpected completed reference root count")
    candidate = json.loads(args.candidate_report.read_text())
    if candidate["status"] != "completed":
        raise ValueError("candidate audit must have completed")
    result = compare_roots([json.loads(path.read_text()) for path in files], candidate["rows"])
    result["schema"] = "reacher_d2p_rgb_protocol_equivalence.v1"
    result["inputs"] = [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for path in [*files, args.candidate_report, Path(__file__)]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "matched_root_count")}))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
