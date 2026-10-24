"""Freeze a P1 margin band against the declared physical task coordinate.

Unlike the RGB-encoder fidelity selector, this calibration-only selector uses
the evaluator's predeclared Reacher qpos-distance mechanism coordinate.  It
never enters the planner and therefore cannot teach, repair, or rank actions.
It only decides whether a model-margin interval is safe enough to test on a
new root-disjoint calibration cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "d2p_reacher_p1_task_critical_margin_band_selection.v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    if report.get("schema") != "d2p_reacher_p1_reconfiguration.v1" or report.get("status") != "calibration_only":
        raise ValueError("task-critical selection requires a P1 calibration-only receipt")
    if report.get("protocol", {}).get("mechanism_observability") != "qpos_distance":
        raise ValueError("task-critical selection requires qpos_distance observability")
    rows: list[dict[str, Any]] = []
    for source in report.get("rows", []):
        first = source.get("methods", {}).get("reconfiguration", {}).get("first_receipt", {})
        physical = source.get("methods", {}).get("reconfiguration", {}).get("physical", {})
        if not first.get("non_greedy_witness_found"):
            continue
        margin = float(first.get("non_greedy_model_margin", float("nan")))
        after = float(physical.get("after_first_chunk_qpos_distance", float("nan")))
        zero = float(physical.get("after_zero_prefix_qpos_distance", float("nan")))
        if not np.isfinite([margin, after, zero]).all() or margin <= 0.0:
            raise ValueError(f"eligible physical calibration root {source.get('seed')} is malformed")
        rows.append({"seed": int(source["seed"]), "model_non_greedy_margin": margin, "candidate_minus_zero_prefix_qpos_distance": after - zero, "preserves_physical_non_greediness": bool(after >= zero - 1.0e-8)})
    if not rows:
        raise ValueError("P1 calibration has no model-eligible roots")
    return rows


def select(report: dict[str, Any], *, minimum_passing_roots: int) -> dict[str, Any]:
    if minimum_passing_roots < 2:
        raise ValueError("task-critical margin band needs at least two roots")
    rows = _rows(report)
    passed = [row for row in rows if row["preserves_physical_non_greediness"]]
    if len(passed) < minimum_passing_roots:
        raise ValueError("insufficient physically non-greedy roots for a task-critical margin band")
    values = sorted({float(row["model_non_greedy_margin"]) for row in passed})
    candidates: list[tuple[list[dict[str, Any]], float, float]] = []
    for offset, low in enumerate(values):
        for high in values[offset:]:
            admitted = [row for row in rows if low <= row["model_non_greedy_margin"] <= high]
            if len(admitted) >= minimum_passing_roots and all(row["preserves_physical_non_greediness"] for row in admitted):
                candidates.append((admitted, low, high))
    if not candidates:
        raise ValueError("no contiguous model-margin interval is physically non-greedy on source calibration")
    admitted, low, high = max(candidates, key=lambda item: (len(item[0]), item[2] - item[1], -item[1], item[2]))
    protocol = report["protocol"]
    return {"schema": SCHEMA, "status": "pass", "selection_rule": "largest calibration-only contiguous closed interval of model-side non-greedy margins with no evaluator-qpos greedy prefix; ties favor support then width; source roots are never outcomes", "selected": {"min_non_greedy_margin": low, "max_non_greedy_margin": high, "calibration_admitted_root_count": len(admitted), "calibration_admitted_root_seeds": [row["seed"] for row in admitted]}, "calibration_rows": rows, "protocol_binding": {"checkpoint_weights_sha256": report.get("inputs", {}).get("weights_sha256"), "adapter_sha256": report.get("inputs", {}).get("adapter_sha256"), "runner_sha256": report.get("inputs", {}).get("runner_sha256"), "mechanism_observability": "qpos_distance", **{key: protocol.get(key) for key in ("reconfiguration_horizon", "recompiled_progress_horizon", "proposals", "proposal_distribution", "selection_metric", "damping", "action_block", "max_relative_reachability_improvement", "min_second_to_first_singular_ratio", "warmup_policy", "target_delta")}}, "claim_boundary": "Calibration-only physical-task abstention rule. qpos is evaluator-only and never reaches the frozen RGB planner; fresh root-disjoint calibration is required before an outcome can open."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--minimum-passing-roots", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"immutable task-critical selection exists: {args.output}")
    report = json.loads(args.report.read_text())
    result = select(report, minimum_passing_roots=args.minimum_passing_roots)
    result["inputs"] = {"report": str(args.report.resolve()), "report_sha256": _sha(args.report)}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "selected": result["selected"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__": main()
