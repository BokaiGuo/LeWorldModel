"""Freeze a P1 reconfiguration margin guard from a calibration-only receipt.

The P1 compiler already emits a model-side immediate non-greedy margin.  This
selector uses a *completed calibration panel only* to determine whether a
positive margin separates roots whose rendered candidate-vs-zero ordering is
preserved from roots where it flips.  It produces an abstention threshold;
fresh roots must still pass a new calibration before any outcome panel opens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "d2p_reacher_p1_margin_guard_selection.v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(*, report: dict[str, Any], fidelity: dict[str, Any], minimum_passing_roots: int) -> dict[str, Any]:
    if minimum_passing_roots < 2:
        raise ValueError("margin calibration requires at least two passing roots")
    if report.get("schema") != "d2p_reacher_p1_reconfiguration.v1" or report.get("status") != "calibration_only":
        raise ValueError("margin selection requires a P1 calibration-only receipt")
    if fidelity.get("schema") != "d2p_reacher_p1_first_chunk_fidelity_audit.v1" or fidelity.get("status") != "completed":
        raise ValueError("margin selection requires a completed first-chunk fidelity receipt")
    fidelity_rows = {int(row["seed"]): row for row in fidelity.get("rows", [])}
    rows: list[dict[str, Any]] = []
    for source in report.get("rows", []):
        method = source.get("methods", {}).get("reconfiguration", {})
        first = method.get("first_receipt", {})
        if not first.get("non_greedy_witness_found"):
            continue
        seed = int(source["seed"])
        audit = fidelity_rows.get(seed)
        if audit is None:
            raise ValueError(f"eligible calibration root {seed} is missing from the fidelity audit")
        margin = float(first.get("non_greedy_model_margin", float("nan")))
        observed_delta = float(audit.get("observed_candidate_minus_zero_goal_cost", float("nan")))
        model_delta = float(audit.get("model_candidate_minus_zero_goal_cost", float("nan")))
        if not np.isfinite([margin, observed_delta, model_delta]).all() or margin <= 0.0:
            raise ValueError(f"eligible calibration root {seed} has invalid margins")
        rows.append({
            "seed": seed,
            "model_non_greedy_margin": margin,
            "observed_candidate_minus_zero_goal_cost": observed_delta,
            "model_candidate_minus_zero_goal_cost": model_delta,
            "model_to_observation_order_agrees": bool(audit.get("model_to_observation_goal_order_agrees")),
            "preserves_non_greedy_order": bool(observed_delta > 0.0 and audit.get("model_to_observation_goal_order_agrees")),
        })
    if not rows:
        raise ValueError("P1 calibration has no model-eligible reconfiguration roots")
    passed = [row for row in rows if row["preserves_non_greedy_order"]]
    failed = [row for row in rows if not row["preserves_non_greedy_order"]]
    if len(passed) < minimum_passing_roots:
        raise ValueError("insufficient calibrated order-preserving roots for a margin guard")
    threshold = min(row["model_non_greedy_margin"] for row in passed)
    admitted = [row for row in rows if row["model_non_greedy_margin"] >= threshold]
    if len(admitted) < minimum_passing_roots or not all(row["preserves_non_greedy_order"] for row in admitted):
        raise ValueError("no positive model-margin threshold yields the required order-preserving calibration subset")
    return {
        "schema": SCHEMA,
        "status": "pass",
        "selection_rule": "smallest strictly positive model non-greedy margin for which every admitted calibration root preserves the rendered candidate-vs-zero non-greedy order; selection roots are never outcomes",
        "selected": {
            "min_non_greedy_margin": float(threshold),
            "calibration_admitted_root_count": len(admitted),
            "calibration_admitted_root_seeds": [row["seed"] for row in admitted],
            "maximum_excluded_failure_margin": max((row["model_non_greedy_margin"] for row in failed), default=None),
        },
        "calibration_rows": rows,
        "protocol_binding": {
            "checkpoint_weights_sha256": report.get("inputs", {}).get("weights_sha256"),
            "adapter_sha256": report.get("inputs", {}).get("adapter_sha256"),
            "runner_sha256": report.get("inputs", {}).get("runner_sha256"),
            "mechanism_observability": report.get("protocol", {}).get("mechanism_observability"),
            "proposal_distribution": report.get("protocol", {}).get("proposal_distribution"),
            "selection_metric": report.get("protocol", {}).get("selection_metric"),
            "damping": report.get("protocol", {}).get("damping"),
            "action_block": report.get("protocol", {}).get("action_block"),
        },
        "claim_boundary": "Calibration-only confidence abstention rule. It neither establishes P1 efficacy nor permits a P1 outcome without a fresh root-disjoint calibration receipt.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-passing-roots", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable margin-guard output exists: {args.output}")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    fidelity = json.loads(args.fidelity.read_text(encoding="utf-8"))
    result = select(report=report, fidelity=fidelity, minimum_passing_roots=args.minimum_passing_roots)
    result["inputs"] = {
        "report": str(args.report.resolve()), "report_sha256": _sha(args.report),
        "fidelity": str(args.fidelity.resolve()), "fidelity_sha256": _sha(args.fidelity),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "selected": result["selected"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
