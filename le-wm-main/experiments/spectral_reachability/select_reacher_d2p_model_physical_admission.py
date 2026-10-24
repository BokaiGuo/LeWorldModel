"""Fail-closed calibration admission for a fresh Reacher D2P P0 outcome panel.

This consumes a *calibration-only* receipt with frozen D2P and CEM settings.
It asks whether the frozen terminal-cost ordering predicts the evaluator-only
physical ordering before any new efficacy roots are opened.  It never selects
actions, damping, or a candidate using the calibration qpos values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sign(value: float, *, atol: float) -> int:
    if not np.isfinite(value):
        raise ValueError("model and physical deltas must be finite")
    return -1 if value < -atol else 1 if value > atol else 0


def _one_sided_sign_p(agreements: int, roots: int) -> float:
    """Exact one-sided sign-test tail under a 50% agreement null."""

    return float(sum(math.comb(roots, count) for count in range(agreements, roots + 1)) / (2 ** roots))


def select(report: dict[str, Any], *, min_roots: int = 8, min_agreements: int = 7, atol: float = 1.0e-9) -> dict[str, Any]:
    if report.get("schema") != "d2p_reacher_p0_goal_progress.v1" or report.get("status") != "calibration_only":
        raise ValueError("admission requires a calibration-only P0 receipt")
    protocol, gates, rows = report.get("protocol", {}), report.get("gates", {}), report.get("rows", [])
    if not protocol.get("calibration_only") or protocol.get("outcome_only"):
        raise ValueError("admission receipt must contain calibration roots only")
    if not gates or not all(gates.values()):
        raise ValueError("admission receipt has failed implementation gates")
    if len(rows) < min_roots or any(row.get("split") != "calibration" for row in rows):
        raise ValueError("admission receipt has insufficient or non-calibration roots")
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        try:
            d2p, cem = row["methods"]["d2p_goal_progress"], row["methods"]["cem"]
            model_delta = float(d2p["predicted_terminal_cost"]) - float(cem["predicted_terminal_cost"])
            physical_delta = float(d2p["physical"]["final_qpos_distance"]) - float(cem["physical"]["final_qpos_distance"])
        except (KeyError, TypeError) as exc:
            raise ValueError("calibration root lacks paired D2P/CEM model and physical costs") from exc
        model_sign, physical_sign = _sign(model_delta, atol=atol), _sign(physical_delta, atol=atol)
        if not model_sign or not physical_sign:
            raise ValueError("calibration admission requires strict paired order on every root")
        comparisons.append({
            "seed": int(row["seed"]),
            "model_d2p_minus_cem_terminal_cost": model_delta,
            "physical_d2p_minus_cem_final_qpos_distance": physical_delta,
            "agrees": bool(model_sign == physical_sign),
        })
    agreement_count = int(sum(row["agrees"] for row in comparisons))
    root_count = len(comparisons)
    sign_p = _one_sided_sign_p(agreement_count, root_count)
    admission = bool(agreement_count >= min_agreements and sign_p <= 0.05)
    return {
        "schema": "d2p_reacher_p0_model_physical_calibration_admission.v1",
        "status": "pass" if admission else "model_physical_calibration_failed",
        "scope": "calibration-only model-to-physical admission; qpos does not select planner actions or tune damping",
        "precommitted_rule": {
            "min_roots": min_roots,
            "min_strict_agreements": min_agreements,
            "max_one_sided_exact_sign_p": 0.05,
        },
        "root_count": root_count,
        "strict_order_agreement_count": agreement_count,
        "strict_order_disagreement_count": root_count - agreement_count,
        "one_sided_exact_sign_p": sign_p,
        "gates": {"receipt_gates_pass": True, "strict_order_every_root": True, "model_physical_admitted": admission},
        "calibration_root_seeds": [row["seed"] for row in comparisons],
        "rows": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    report_path = args.report.resolve()
    result = select(json.loads(report_path.read_text(encoding="utf-8")))
    result["source_report"] = str(report_path)
    result["source_report_sha256"] = _sha256(report_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "root_count", "strict_order_agreement_count", "one_sided_exact_sign_p")}, sort_keys=True))
    if result["status"] != "pass":
        raise RuntimeError("model-to-physical calibration admission failed; do not open P0 outcome roots")


if __name__ == "__main__":
    main()
