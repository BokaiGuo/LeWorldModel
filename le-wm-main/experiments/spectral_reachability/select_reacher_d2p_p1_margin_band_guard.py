"""Select a model-only P1 margin *band* from a sealed calibration receipt.

This is an abstention selector, not an efficacy reducer.  When a lower-bound
margin guard still admits calibration roots whose rendered successor reverses
the frozen-model ordering, it searches only contiguous intervals of the
model-side margin.  The selected band must subsequently pass a fresh,
root-disjoint CUDA calibration before an outcome panel can open.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "d2p_reacher_p1_margin_band_guard_selection.v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(report: dict[str, Any], fidelity: dict[str, Any]) -> list[dict[str, Any]]:
    if report.get("schema") != "d2p_reacher_p1_reconfiguration.v1" or report.get("status") != "calibration_only":
        raise ValueError("margin-band selection requires a P1 calibration-only receipt")
    if fidelity.get("schema") != "d2p_reacher_p1_first_chunk_fidelity_audit.v1" or fidelity.get("status") != "completed":
        raise ValueError("margin-band selection requires a completed first-chunk fidelity receipt")
    binding = fidelity.get("source_binding", {})
    expected_binding = {
        "checkpoint_weights_sha256": report.get("inputs", {}).get("weights_sha256"),
        "adapter_sha256": report.get("inputs", {}).get("adapter_sha256"),
        "target_delta": report.get("protocol", {}).get("target_delta"),
    }
    for key, expected in expected_binding.items():
        if binding.get(key) != expected:
            raise ValueError(f"first-chunk fidelity receipt is incompatible on {key}")
    audits = {int(row["seed"]): row for row in fidelity.get("rows", [])}
    rows: list[dict[str, Any]] = []
    for source in report.get("rows", []):
        first = source.get("methods", {}).get("reconfiguration", {}).get("first_receipt", {})
        if not first.get("non_greedy_witness_found"):
            continue
        seed = int(source["seed"])
        audit = audits.get(seed)
        if audit is None:
            raise ValueError(f"eligible calibration root {seed} is missing from the fidelity audit")
        margin = float(first.get("non_greedy_model_margin", float("nan")))
        model_delta = float(audit.get("model_candidate_minus_zero_goal_cost", float("nan")))
        observed_delta = float(audit.get("observed_candidate_minus_zero_goal_cost", float("nan")))
        if not np.isfinite([margin, model_delta, observed_delta]).all() or margin <= 0.0:
            raise ValueError(f"eligible calibration root {seed} has invalid margins")
        # The physical replay's model score should reproduce the compiler's
        # receipt up to single-precision render/encode noise.  Without this
        # check, a band could be selected from a stale or differently scored
        # packet even while the replay itself succeeds.
        if not np.isclose(margin, model_delta, rtol=0.0, atol=2.0e-6):
            raise ValueError(f"eligible calibration root {seed} has inconsistent compiler and replay model margins")
        agrees = bool(audit.get("model_to_observation_goal_order_agrees"))
        rows.append({
            "seed": seed,
            "model_non_greedy_margin": margin,
            "model_candidate_minus_zero_goal_cost": model_delta,
            "observed_candidate_minus_zero_goal_cost": observed_delta,
            "model_to_observation_goal_order_agrees": agrees,
            "preserves_non_greedy_order": bool(agrees and observed_delta > 0.0),
        })
    if not rows:
        raise ValueError("P1 calibration has no model-eligible reconfiguration roots")
    return rows


def select(*, report: dict[str, Any], fidelity: dict[str, Any], minimum_passing_roots: int) -> dict[str, Any]:
    if minimum_passing_roots < 2:
        raise ValueError("margin-band calibration requires at least two passing roots")
    rows = _rows(report, fidelity)
    passed = [row for row in rows if row["preserves_non_greedy_order"]]
    if len(passed) < minimum_passing_roots:
        raise ValueError("insufficient calibrated order-preserving roots for a margin band")

    # Only observed model margins may become interval endpoints.  Enumerating
    # the finite lattice of closed intervals makes selection deterministic and
    # forbids choosing a continuous cutoff after inspecting a future outcome.
    values = sorted({float(row["model_non_greedy_margin"]) for row in passed})
    candidates: list[tuple[list[dict[str, Any]], float, float]] = []
    for lower_index, lower in enumerate(values):
        for upper in values[lower_index:]:
            admitted = [
                row for row in rows
                if lower <= float(row["model_non_greedy_margin"]) <= upper
            ]
            if len(admitted) >= minimum_passing_roots and all(row["preserves_non_greedy_order"] for row in admitted):
                candidates.append((admitted, lower, upper))
    if not candidates:
        raise ValueError("no contiguous model-margin band yields the required order-preserving calibration subset")

    # Prefer support over width: the selected band is the largest calibration
    # subset with zero observed reversals.  Width and endpoint order only make
    # otherwise tied choices deterministic.
    admitted, lower, upper = max(
        candidates,
        key=lambda item: (len(item[0]), item[2] - item[1], -item[1], item[2]),
    )
    return {
        "schema": SCHEMA,
        "status": "pass",
        "selection_rule": "largest calibration-only contiguous closed interval of compiler-side non-greedy margins containing at least the required number of roots and no rendered candidate-vs-zero order reversals; ties favor wider intervals then lower endpoints; selected roots are never outcomes",
        "selected": {
            "min_non_greedy_margin": float(lower),
            "max_non_greedy_margin": float(upper),
            "calibration_admitted_root_count": len(admitted),
            "calibration_admitted_root_seeds": [row["seed"] for row in admitted],
        },
        "calibration_rows": rows,
        "protocol_binding": {
            "checkpoint_weights_sha256": report.get("inputs", {}).get("weights_sha256"),
            "adapter_sha256": report.get("inputs", {}).get("adapter_sha256"),
            "runner_sha256": report.get("inputs", {}).get("runner_sha256"),
            "mechanism_observability": report.get("protocol", {}).get("mechanism_observability"),
            "reconfiguration_horizon": report.get("protocol", {}).get("reconfiguration_horizon"),
            "recompiled_progress_horizon": report.get("protocol", {}).get("recompiled_progress_horizon"),
            "proposals": report.get("protocol", {}).get("proposals"),
            "proposal_distribution": report.get("protocol", {}).get("proposal_distribution"),
            "selection_metric": report.get("protocol", {}).get("selection_metric"),
            "damping": report.get("protocol", {}).get("damping"),
            "action_block": report.get("protocol", {}).get("action_block"),
            "max_relative_reachability_improvement": report.get("protocol", {}).get("max_relative_reachability_improvement"),
            "min_second_to_first_singular_ratio": report.get("protocol", {}).get("min_second_to_first_singular_ratio"),
            "warmup_policy": report.get("protocol", {}).get("warmup_policy"),
            "target_delta": report.get("protocol", {}).get("target_delta"),
        },
        "claim_boundary": "Calibration-only model-confidence abstention rule. The band has no efficacy claim and requires a fresh root-disjoint calibration receipt before a P1 outcome can open.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-passing-roots", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable margin-band output exists: {args.output}")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    fidelity = json.loads(args.fidelity.read_text(encoding="utf-8"))
    if fidelity.get("source_report_sha256") != _sha(args.report):
        raise ValueError("first-chunk fidelity receipt is bound to a different calibration report")
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
