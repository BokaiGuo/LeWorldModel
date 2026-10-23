"""Audit whether V6 pre-outcome diagnostics could have routed endpoint repair.

The V6 endpoint model was properly selected and certified on independent
calibration roots, but its mean regret regressed on a sealed outcome panel.
This audit separates two explanations:

1. a calibration-selected, pre-outcome diagnostic could have routed some roots
   to the cost carrier and avoided the regression; or
2. the available endpoint margin/ensemble diagnostics do not transfer, so a
   passive uncertainty gate is not a credible repairability mechanism.

It is intentionally a *post-execution mechanism audit*.  The source report
already contains outcome labels, so its result cannot authorize a deployment
policy.  It does, however, use only pre-outcome fields to construct every
candidate route and retains V6's selector/certification root partition before
looking at the outcome section.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


SCHEMA = "pointmaze_v6_prospective_routing_post_execution_audit.v1"
FEATURES = (
    "endpoint_margin",
    "cost_margin",
    "endpoint_vs_cost_score_gap",
    "ensemble_member_agreement",
    "ensemble_distinct_winners",
    "ensemble_score_standard_deviation",
    "ensemble_selected_xy_spread",
)
QUANTILES = tuple(float(value) for value in np.arange(0.10, 0.91, 0.10))
MINIMUM_ROUTE_ROOTS = 12


@dataclass(frozen=True)
class RootRecord:
    root: str
    features: dict[str, float]
    cost_regret: float
    endpoint_regret: float


@dataclass(frozen=True)
class Route:
    name: str
    feature: str | None
    direction: str | None
    threshold: float | None


def _role(root: str) -> int:
    """Match V6's immutable selector/certification/risk root allocation."""

    return int(hashlib.sha256(root.encode("utf-8")).hexdigest()[:8], 16) % 3


def _margin(values: Any, *, name: str) -> float:
    scores = np.sort(np.asarray(values, dtype=np.float64))
    if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError(f"{name} must contain at least two finite candidate scores")
    return float(scores[1] - scores[0])


def _validate_report(payload: dict[str, Any]) -> None:
    if payload.get("schema") != "pointmaze_counterfactual_endpoint_control.v1":
        raise ValueError("audit requires a completed V6 endpoint-control report")
    if payload.get("status") != "completed_counterfactual_endpoint_control":
        raise ValueError("audit requires completed V6 endpoint-control evidence")
    contract = payload.get("control_contract")
    if not isinstance(contract, dict) or contract.get("endpoint_ensemble_disagreement_diagnostic_only") is not True:
        raise ValueError("report does not establish the V6 pre-outcome ensemble diagnostic contract")
    if not isinstance(payload.get("calibration", {}).get("per_root"), dict):
        raise ValueError("report lacks calibration per-root records")
    if not isinstance(payload.get("outcome", {}).get("per_root"), dict):
        raise ValueError("report lacks outcome per-root records")


def _records(rows: dict[str, Any], diagnostic: dict[str, Any]) -> dict[str, RootRecord]:
    if diagnostic.get("available") is not True:
        raise ValueError("V6 endpoint ensemble diagnostic is unavailable")
    per_root = diagnostic.get("per_root")
    if not isinstance(per_root, dict) or set(per_root) != set(rows):
        raise ValueError("endpoint diagnostic roots do not exactly match evaluation roots")
    result: dict[str, RootRecord] = {}
    for root, row in sorted(rows.items()):
        if not isinstance(row, dict) or not isinstance(per_root[root], dict):
            raise ValueError(f"{root}: invalid per-root report")
        endpoint = np.asarray(row.get("endpoint_predicted_cost"), dtype=np.float64)
        cost = np.asarray(row.get("cost_predicted_cost"), dtype=np.float64)
        candidate_ids = [str(value) for value in row.get("candidate_ids", [])]
        if len(candidate_ids) != len(endpoint) or len(endpoint) != len(cost):
            raise ValueError(f"{root}: candidate scores and identities do not align")
        endpoint_choice = str(row.get("endpoint_candidate_id"))
        cost_choice = str(row.get("cost_candidate_id"))
        if endpoint_choice not in candidate_ids or cost_choice not in candidate_ids:
            raise ValueError(f"{root}: selected candidate is absent from fixed bank")
        endpoint_index, cost_index = candidate_ids.index(endpoint_choice), candidate_ids.index(cost_choice)
        ensemble = per_root[root]
        features = {
            "endpoint_margin": _margin(endpoint, name=f"{root} endpoint"),
            "cost_margin": _margin(cost, name=f"{root} cost"),
            # Positive values mean the endpoint model gives its own candidate a
            # lower score than the cost model's candidate.  This uses neither
            # realised successor nor task cost.
            "endpoint_vs_cost_score_gap": float(endpoint[cost_index] - endpoint[endpoint_index]),
            "ensemble_member_agreement": float(ensemble["member_agreement_with_deployed_choice"]),
            "ensemble_distinct_winners": float(ensemble["distinct_member_winner_count"]),
            "ensemble_score_standard_deviation": float(ensemble["mean_candidate_score_standard_deviation"]),
            "ensemble_selected_xy_spread": float(ensemble["selected_endpoint_xy_spread_mean"]),
        }
        if set(features) != set(FEATURES) or not np.isfinite(list(features.values())).all():
            raise ValueError(f"{root}: non-finite pre-outcome routing feature")
        result[root] = RootRecord(
            root=root,
            features=features,
            cost_regret=float(row["cost_regret"]),
            endpoint_regret=float(row["endpoint_regret"]),
        )
    return result


def _route_endpoint(route: Route, record: RootRecord) -> bool:
    if route.name == "cost_only":
        return False
    if route.name == "endpoint_only":
        return True
    if route.feature is None or route.direction is None or route.threshold is None:
        raise ValueError("conditional route lacks a complete fixed predicate")
    value = float(record.features[route.feature])
    if route.direction == "high":
        return value >= float(route.threshold)
    if route.direction == "low":
        return value <= float(route.threshold)
    raise ValueError(f"unsupported route direction {route.direction!r}")


def _route_summary(route: Route, records: dict[str, RootRecord]) -> dict[str, Any]:
    endpoint = [record for record in records.values() if _route_endpoint(route, record)]
    cost = [record for record in records.values() if not _route_endpoint(route, record)]
    regret = [record.endpoint_regret if _route_endpoint(route, record) else record.cost_regret for record in records.values()]
    delta = [
        (record.endpoint_regret if _route_endpoint(route, record) else record.cost_regret) - record.cost_regret
        for record in records.values()
    ]
    return {
        "route": {"name": route.name, "feature": route.feature, "direction": route.direction, "threshold": route.threshold},
        "root_count": len(records),
        "endpoint_root_count": len(endpoint),
        "cost_root_count": len(cost),
        "endpoint_coverage": float(len(endpoint) / len(records)) if records else 0.0,
        "mean_regret": float(np.mean(regret)) if regret else None,
        "mean_delta_vs_cost": float(np.mean(delta)) if delta else None,
    }


def _routes(selector: dict[str, RootRecord]) -> list[Route]:
    if not selector:
        raise ValueError("selector roots are required")
    result = [Route("cost_only", None, None, None), Route("endpoint_only", None, None, None)]
    for feature in FEATURES:
        values = np.asarray([record.features[feature] for record in selector.values()], dtype=np.float64)
        for threshold in sorted(set(float(np.quantile(values, value)) for value in QUANTILES)):
            for direction in ("low", "high"):
                result.append(Route("threshold", feature, direction, threshold))
    return result


def _eligible(summary: dict[str, Any]) -> bool:
    endpoint, cost = int(summary["endpoint_root_count"]), int(summary["cost_root_count"])
    return bool(
        summary["route"]["name"] in {"cost_only", "endpoint_only"}
        or (endpoint >= MINIMUM_ROUTE_ROOTS and cost >= MINIMUM_ROUTE_ROOTS)
    )


def _bootstrap_upper_delta(records: dict[str, RootRecord], route: Route, *, draws: int, seed: int) -> float:
    values = np.asarray([
        (record.endpoint_regret if _route_endpoint(route, record) else record.cost_regret) - record.cost_regret
        for record in records.values()
    ], dtype=np.float64)
    if not len(values):
        raise ValueError("certification requires at least one root")
    rng = np.random.default_rng(int(seed))
    sample = values[rng.integers(0, len(values), size=(int(draws), len(values)))].mean(axis=1)
    return float(np.quantile(sample, 0.95))


def audit(payload: dict[str, Any], *, bootstrap_draws: int = 20_000, seed: int = 202610350) -> dict[str, Any]:
    """Run the frozen V6 selector/certification decomposition post execution."""

    _validate_report(payload)
    calibration_rows = _records(
        payload["calibration"]["per_root"], payload["endpoint_ensemble_diagnostics_pre_outcome"]["calibration"]
    )
    outcome_rows = _records(
        payload["outcome"]["per_root"], payload["endpoint_ensemble_diagnostics_pre_outcome"]["outcome"]
    )
    selector = {root: row for root, row in calibration_rows.items() if _role(root) == 0}
    certification = {root: row for root, row in calibration_rows.items() if _role(root) == 1}
    risk = {root: row for root, row in calibration_rows.items() if _role(root) == 2}
    if not selector or not certification or not risk:
        raise ValueError("V6 role partition has an empty calibration role")
    candidates = [_route_summary(route, selector) for route in _routes(selector)]
    eligible = [candidate for candidate in candidates if _eligible(candidate)]
    if not eligible:
        raise ValueError("no route met predeclared selector support")
    chosen_summary = min(
        eligible,
        key=lambda row: (
            float(row["mean_regret"]),
            -min(int(row["endpoint_root_count"]), int(row["cost_root_count"])),
            json.dumps(row["route"], sort_keys=True),
        ),
    )
    route_payload = chosen_summary["route"]
    chosen = Route(
        str(route_payload["name"]), route_payload["feature"], route_payload["direction"], route_payload["threshold"]
    )
    certification_summary = _route_summary(chosen, certification)
    certification_upper = _bootstrap_upper_delta(certification, chosen, draws=int(bootstrap_draws), seed=int(seed))
    authorized = bool(certification_summary["mean_delta_vs_cost"] <= 0.0 and certification_upper <= 0.0)
    outcome_summary = _route_summary(chosen, outcome_rows)
    outcome_upper = _bootstrap_upper_delta(outcome_rows, chosen, draws=int(bootstrap_draws), seed=int(seed) + 1)
    return {
        "schema": SCHEMA,
        "status": "completed_post_execution_prospective_routing_audit",
        "claim_boundary": (
            "Post-execution mechanism audit of a V6 report whose outcome labels are already materialized. "
            "It cannot authorize deployment or establish a prospective repairability result; it only determines "
            "whether fixed pre-outcome diagnostics have enough evidence to justify a fresh-panel routing test."
        ),
        "protocol": {
            "base_report_schema": payload["schema"],
            "feature_fields_are_pre_outcome": list(FEATURES),
            "selector": "V6 calibration role 0 selects one route from fixed univariate predicates",
            "certification": "V6 calibration role 1 uses a paired root bootstrap upper delta <= 0",
            "risk_calibration": "V6 calibration role 2 is held out from route selection/certification",
            "outcome": "read only after route and certification are fixed",
            "conditional_minimum_roots_per_arm": MINIMUM_ROUTE_ROOTS,
            "quantiles_from_selector_only": list(QUANTILES),
        },
        "root_counts": {"selector": len(selector), "certification": len(certification), "risk_calibration": len(risk), "outcome": len(outcome_rows)},
        "selector": {
            "candidate_route_count": len(candidates),
            "eligible_route_count": len(eligible),
            "chosen": chosen_summary,
        },
        "certification": {
            **certification_summary,
            "upper_95_route_minus_cost_delta": certification_upper,
            "would_authorize_on_calibration": authorized,
        },
        "outcome": {
            **outcome_summary,
            "upper_95_route_minus_cost_delta": outcome_upper,
            "would_improve_over_cost": bool(outcome_summary["mean_delta_vs_cost"] <= 0.0 and outcome_upper <= 0.0),
        },
        "decision": {
            "fresh_panel_route_warranted": bool(authorized and outcome_summary["mean_delta_vs_cost"] <= 0.0 and outcome_upper <= 0.0),
            "interpretation": (
                "The calibration-selected pre-outcome route transferred in this already-opened audit. A new root/action-disjoint panel is warranted to test it prospectively."
                if authorized and outcome_summary["mean_delta_vs_cost"] <= 0.0 and outcome_upper <= 0.0
                else "The available V6 pre-outcome univariate diagnostics did not yield a calibration-authorized route with stable outcome improvement. Do not promote an uncertainty gate; prioritize a new information/probe intervention."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=202610350)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite post-execution audit: {args.output}")
    if args.bootstrap_draws < 1000:
        raise ValueError("bootstrap-draws must be at least 1000")
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    result = audit(payload, bootstrap_draws=int(args.bootstrap_draws), seed=int(args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "decision": result["decision"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
