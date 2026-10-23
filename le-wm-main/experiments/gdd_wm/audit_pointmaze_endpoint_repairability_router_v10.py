"""Audit V10's immutable held-out consequence without reselecting a policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "pointmaze_endpoint_repairability_router_v10_outcome_audit.v1"
_EPS = 1.0e-10


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _upper_bootstrap(delta: np.ndarray, *, seed: int, draws: int = 4096) -> float:
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError("paired outcome delta must be a finite non-empty vector")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(delta), size=(int(draws), len(delta)))
    return float(np.quantile(delta[indices].mean(axis=1), 0.95))


def audit(report_path: Path, *, seed: int = 202610405) -> dict[str, Any]:
    report = _load(report_path)
    if (
        report.get("schema") != "pointmaze_endpoint_repairability_router_v10.v1"
        or report.get("status") != "completed_execution_grounded_endpoint_repairability_router"
    ):
        raise ValueError("requires a V10 endpoint-repairability report")
    contract = report.get("control_contract")
    routing = report.get("calibration", {}).get("routing")
    rows = report.get("outcome", {}).get("per_root")
    if not isinstance(contract, dict) or not isinstance(routing, dict) or not isinstance(rows, dict) or not rows:
        raise ValueError("V10 report lacks control contract, frozen routing policy, or outcome roots")
    if (
        contract.get("repair_arm") != "endpoint_coupled_counterfactual"
        or contract.get("fallback_arm") != "direct_cost_only"
        or contract.get("same_choice_roots_forced_to_cost") is not True
        or contract.get("outcome_execution_deferred_until_router_threshold_and_certificate_frozen") is not True
        or contract.get("outcome_cost_used_for_router_or_threshold_selection") is not False
    ):
        raise ValueError("V10 control contract is incomplete or permits outcome leakage")
    policy = routing.get("deployment_policy")
    if policy not in {"cost", "routed"}:
        raise ValueError("V10 calibration lacks a valid frozen deployment policy")
    if policy == "routed" and routing.get("routing_authorized") is not True:
        raise ValueError("V10 routed deployment lacks its calibration authorization")
    delta: list[float] = []
    release_count = 0
    violations: dict[str, str] = {}
    for root, row in sorted(rows.items()):
        if not isinstance(row, dict):
            raise ValueError(f"{root}: outcome row must be an object")
        ids = row.get("candidate_ids")
        cost_id, endpoint_id, routed_id = row.get("cost_candidate_id"), row.get("endpoint_candidate_id"), row.get("routed_candidate_id")
        eligible = bool(row.get("routing_eligible"))
        disagreement = bool(row.get("candidate_disagreement"))
        released = bool(row.get("released_endpoint"))
        regrets = {name: float(row.get(f"{name}_regret", np.nan)) for name in ("cost", "endpoint", "routed")}
        if not isinstance(ids, list) or cost_id not in ids or endpoint_id not in ids or routed_id not in ids:
            violations[str(root)] = "candidate identity is incomplete or incoherent"
        elif eligible != disagreement or eligible != (cost_id != endpoint_id):
            violations[str(root)] = "declared opportunity does not equal scorer candidate disagreement"
        elif not eligible and (released or routed_id != cost_id or abs(regrets["routed"] - regrets["cost"]) > _EPS):
            violations[str(root)] = "same-choice root escaped the required cost fallback"
        elif policy == "cost" and (released or routed_id != cost_id or abs(regrets["routed"] - regrets["cost"]) > _EPS):
            violations[str(root)] = "unauthorized router did not preserve the cost fallback"
        elif policy == "routed" and (routed_id != (endpoint_id if released else cost_id) or abs(regrets["routed"] - (regrets["endpoint"] if released else regrets["cost"])) > _EPS):
            violations[str(root)] = "routed action/regret is inconsistent with the frozen release flag"
        elif not np.isfinite(list(regrets.values())).all():
            violations[str(root)] = "non-finite regret"
        delta.append(regrets["routed"] - regrets["cost"])
        release_count += int(released)
    if violations:
        raise ValueError(f"V10 outcome routing contract violated: {violations}")
    values = np.asarray(delta, dtype=np.float64)
    mean = float(values.mean())
    upper = _upper_bootstrap(values, seed=seed)
    if policy == "routed":
        status = "validated_held_out_endpoint_repair_gain" if mean < 0.0 and upper <= 0.0 and release_count else "held_out_endpoint_repair_not_confirmed"
    else:
        status = "validated_fail_closed_cost_fallback"
    return {
        "schema": SCHEMA, "status": status,
        "report": str(Path(report_path).resolve()), "report_sha256": _sha256(report_path),
        "frozen_deployment_policy": policy, "routing_authorized_on_calibration": bool(routing.get("routing_authorized")),
        "outcome_root_count": len(rows), "outcome_endpoint_release_count": release_count,
        "outcome_endpoint_release_coverage": float(release_count / len(rows)),
        "outcome_mean_routed_minus_cost_regret": mean,
        "outcome_upper_95_routed_minus_cost_regret": upper,
        "success_criterion": "authorized routed policy, nonzero release, held-out paired mean and upper-95 delta below zero",
        "claim_boundary": "Fixed-policy held-out PointMaze candidate-bank audit. It neither reselects a threshold nor establishes conditional coverage, online MPC, or a universal repairability result.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=202610405)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable V10 outcome audit: {args.output}")
    result = audit(args.report, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "outcome_mean_routed_minus_cost_regret", "outcome_upper_95_routed_minus_cost_regret")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
