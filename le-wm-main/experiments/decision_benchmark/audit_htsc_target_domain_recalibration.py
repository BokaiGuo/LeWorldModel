"""Evaluate a predeclared target-domain HTSC recalibration split.

The source-domain fixed-bank certificate can fail under root-distribution
shift.  This audit does not reinterpret a failed transport certificate: it
models a different deployment contract in which a small, chronologically
declared target-domain calibration panel is labeled before later target roots
are encountered.  Calibration and evaluation are split by root, and root
maximum contrast errors remain the only calibration statistic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark.task_contrast import _wilson_interval, calibrate_root_score_bound


SOURCE_SCHEMA = "htsc.finite_sample_calibration_audit.v1"
SCHEMA = "htsc.target_domain_recalibration_audit.v1"


def target_domain_recalibration(
    source_audit: dict[str, Any], *,
    target_calibration_root_count: int,
    alpha: float,
) -> dict[str, Any]:
    """Freeze a chronological target-domain recalibration before later roots.

    The source artifact is expected to contain one maximum contrast error per
    target root.  The chosen prefix is a deployment-time calibration set; the
    suffix is untouched until the threshold has been frozen.  This makes the
    result a valid counterfactual *target-recalibration* protocol, while
    retaining the original source-to-target failure as separate evidence.
    """
    if source_audit.get("schema") != SOURCE_SCHEMA:
        raise ValueError("source audit must be a finite-sample HTSC audit")
    roots = source_audit.get("independent_external_root_coverage", {}).get("roots")
    if not isinstance(roots, list) or len(roots) < 3:
        raise ValueError("source audit requires at least three target-root records")
    count = int(target_calibration_root_count)
    if count < 1 or count >= len(roots):
        raise ValueError("target calibration root count must leave at least one evaluation root")
    ordered: list[tuple[str, float]] = []
    for row in roots:
        root, error = str(row.get("group_id", "")), row.get("max_contrast_error")
        if not root or not np.isfinite(float(error)) or float(error) < 0.0:
            raise ValueError("source audit has invalid target root/error records")
        ordered.append((root, float(error)))
    if len({root for root, _ in ordered}) != len(ordered):
        raise ValueError("source audit target roots must be unique")
    ordered.sort(key=lambda item: item[0])
    calibration, outcome = ordered[:count], ordered[count:]
    bound = calibrate_root_score_bound(
        np.asarray([error for _, error in calibration], dtype=np.float64), alpha=float(alpha),
    )
    if bound["bound"] is None:
        covered_count = None
        coverage = None
        interval = None
    else:
        covered_count = int(sum(error <= float(bound["bound"]) for _, error in outcome))
        coverage = float(covered_count / len(outcome))
        interval = _wilson_interval(covered_count, len(outcome))
    return {
        "schema": SCHEMA,
        "status": "evaluated" if bound["bound"] is not None else "fail_closed_insufficient_target_calibration",
        "source_transport_audit_schema": SOURCE_SCHEMA,
        "target_split_contract": {
            "rule": "lexicographically ordered target root IDs; prefix is calibration and suffix is outcome",
            "target_calibration_root_count": len(calibration),
            "target_outcome_root_count": len(outcome),
            "outcome_root_errors_read_after_target_threshold_freeze": True,
            "selection_uses_target_outcome_errors": False,
        },
        "target_calibration": {
            **bound,
            "root_ids": [root for root, _ in calibration],
        },
        "target_outcome": {
            "root_ids": [root for root, _ in outcome],
            "root_max_contrast_errors": [error for _, error in outcome],
            "covered_root_count": covered_count,
            "empirical_root_coverage": coverage,
            "empirical_root_coverage_wilson_95_interval": (
                None if interval is None else {"lower": interval[0], "upper": interval[1]}
            ),
        },
        "claim_boundary": (
            "Counterfactual target-domain sequential recalibration for one fixed candidate bank. "
            "It does not repair the original source-to-target certificate, establish conditional coverage, "
            "authorize adaptive candidates, or demonstrate closed-loop utility."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-calibration-root-count", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite target-domain recalibration artifact")
    source = json.loads(args.source_audit.read_text(encoding="utf-8"))
    result = target_domain_recalibration(
        source, target_calibration_root_count=args.target_calibration_root_count, alpha=args.alpha,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "artifact": str(args.output), "status": result["status"],
        "target_bound": result["target_calibration"]["bound"],
        "outcome_coverage": result["target_outcome"]["empirical_root_coverage"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
