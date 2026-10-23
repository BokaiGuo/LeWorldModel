"""Decompose where a planning-shaped repairability control can actually help.

The V5 repairability report contains a cost-only scorer, a planning-shaped
scorer, and a calibration-frozen prospective router.  A panel mean alone is
misleading: when both scoring arms pick the same candidate, no routing method
can create a decision improvement.  This post-execution audit therefore
reports, separately for calibration and outcome roots:

* scorer disagreement (the repairable decision stratum);
* oracle candidate-bank headroom under the cost scorer;
* realized planning-shaped improvement conditional on disagreement;
* the frozen router's precision and recall for those improvements;
* released-arm regret and missed improvements.

It never selects a threshold, changes an authorized policy, or turns an
opened panel into prospective evidence.  A future fresh panel is required to
test whether any observed diagnostic supports a repair policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "pointmaze_planning_shaped_repairability_mechanism_audit.v1"
_EPSILON = 1.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rate(mask: np.ndarray) -> float | None:
    return None if not len(mask) else float(np.mean(mask))


def _mean(values: np.ndarray) -> float | None:
    return None if not len(values) else float(np.mean(values))


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _partition(
    rows: dict[str, Any], *, name: str, opportunity_gated: bool | None,
) -> dict[str, Any]:
    if not isinstance(rows, dict) or not rows:
        raise ValueError(f"{name}: per_root must be a non-empty object")

    parsed: list[dict[str, Any]] = []
    for root, row in sorted(rows.items()):
        if not isinstance(row, dict):
            raise ValueError(f"{name}/{root}: result must be an object")
        values = np.asarray([row.get("cost_regret"), row.get("planning_regret"), row.get("routed_regret")], dtype=np.float64)
        cost_candidate = row.get("cost_candidate_id")
        planning_candidate = row.get("planning_candidate_id")
        released = row.get("released_planning")
        if cost_candidate is None or planning_candidate is None or not isinstance(released, bool) or not np.isfinite(values).all():
            raise ValueError(f"{name}/{root}: missing finite regret, candidate identity, or release flag")
        improvement = bool(values[1] < values[0] - _EPSILON)
        declared = row.get("planning_beats_cost")
        if declared is not None and declared is not improvement:
            raise ValueError(f"{name}/{root}: planning_beats_cost disagrees with regrets")
        if released:
            expected_routed = values[1]
        else:
            expected_routed = values[0]
        if not np.isclose(values[2], expected_routed, rtol=0.0, atol=_EPSILON):
            raise ValueError(f"{name}/{root}: routed regret does not match frozen release decision")
        disagreement = str(cost_candidate) != str(planning_candidate)
        # V9 writes these fields at decision time.  Validate them here rather
        # than trusting a descriptive outcome flag: otherwise a same-choice
        # root could be incorrectly counted as a route opportunity after its
        # realized regret is known.  Historical V5 reports omit the fields and
        # remain readable as historical mechanism audits.
        declared_disagreement = row.get("candidate_disagreement")
        if declared_disagreement is not None:
            if not isinstance(declared_disagreement, bool) or declared_disagreement is not disagreement:
                raise ValueError(f"{name}/{root}: candidate_disagreement disagrees with candidate identities")
        declared_eligible = row.get("routing_eligible")
        if declared_eligible is not None:
            if not isinstance(declared_eligible, bool):
                raise ValueError(f"{name}/{root}: routing_eligible must be boolean")
            # V9's explicit opportunity gate constrains routing to scorer
            # disagreements.  V7 deliberately has no such gate: every root
            # is eligible for its advantage router, while same-choice roots
            # merely have zero decision opportunity.  Historical reports
            # without a control contract retain the old strict interpretation.
            requires_disagreement = opportunity_gated is not False
            if requires_disagreement and declared_eligible is not disagreement:
                raise ValueError(f"{name}/{root}: routing_eligible must equal scorer candidate disagreement")
            if not requires_disagreement and not declared_eligible:
                raise ValueError(f"{name}/{root}: non-gated router must mark every root routing-eligible")
            if requires_disagreement and not declared_eligible and released:
                raise ValueError(f"{name}/{root}: opportunity-gated router released a same-choice root")
        parsed.append({
            "root": str(root), "cost_candidate": str(cost_candidate), "planning_candidate": str(planning_candidate),
            "cost_regret": float(values[0]), "planning_regret": float(values[1]), "routed_regret": float(values[2]),
            "disagreement": disagreement, "improvement": improvement,
            "released": released, "has_eligibility_field": declared_eligible is not None,
        })

    disagreement = np.asarray([row["disagreement"] for row in parsed], dtype=bool)
    improvement = np.asarray([row["improvement"] for row in parsed], dtype=bool)
    released = np.asarray([row["released"] for row in parsed], dtype=bool)
    cost_regret = np.asarray([row["cost_regret"] for row in parsed], dtype=np.float64)
    planning_regret = np.asarray([row["planning_regret"] for row in parsed], dtype=np.float64)
    routed_regret = np.asarray([row["routed_regret"] for row in parsed], dtype=np.float64)
    effective_improvement = disagreement & improvement
    released_effective_improvement = released & effective_improvement
    released_regret_delta = planning_regret[released] - cost_regret[released]
    missed = (~released) & effective_improvement
    eligibility_fields = np.asarray([row["has_eligibility_field"] for row in parsed], dtype=bool)

    return {
        "root_count": len(parsed),
        "candidate_choice": {
            "disagreement_count": int(disagreement.sum()),
            "disagreement_rate": _rate(disagreement),
            "same_choice_count": int((~disagreement).sum()),
            "disagreement_roots": [row["root"] for row in parsed if row["disagreement"]],
        },
        "opportunity_gate": {
            "enabled": opportunity_gated is True,
            "decision_time_contract_present": bool(eligibility_fields.any()),
            "decision_time_contract_root_count": int(eligibility_fields.sum()),
            "eligible_root_count": (
                int(disagreement.sum()) if opportunity_gated is not False
                else len(parsed)
            ) if eligibility_fields.any() else None,
            "same_choice_release_count": int((released & ~disagreement).sum()),
            "claim_boundary": "When present, these are consistency checks against pre-outcome decision fields; they do not make the post-execution audit a new routing authorization.",
        },
        "candidate_bank_headroom": {
            "cost_oracle_headroom_mean": _mean(cost_regret),
            "cost_oracle_headroom_mean_when_disagree": _mean(cost_regret[disagreement]),
            "positive_headroom_rate_when_disagree": _rate(cost_regret[disagreement] > _EPSILON),
            "claim_boundary": "Regret is oracle headroom within the already-executed fixed candidate bank; it neither certifies the bank nor measures reachable closed-loop task improvement.",
        },
        "planning_shaped_effect": {
            "strict_improvement_count": int(improvement.sum()),
            "strict_improvement_rate": _rate(improvement),
            "strict_improvement_count_when_disagree": int(effective_improvement.sum()),
            "strict_improvement_rate_when_disagree": _rate(improvement[disagreement]),
            "mean_planning_minus_cost_regret_when_disagree": _mean((planning_regret - cost_regret)[disagreement]),
        },
        "frozen_router": {
            "release_count": int(released.sum()),
            "release_rate": _rate(released),
            "effective_release_count": int(released_effective_improvement.sum()),
            "precision_for_effective_improvement": _safe_ratio(int(released_effective_improvement.sum()), int(released.sum())),
            "recall_for_effective_improvement": _safe_ratio(int(released_effective_improvement.sum()), int(effective_improvement.sum())),
            "released_arm_mean_planning_minus_cost_regret": _mean(released_regret_delta),
            "all_root_mean_routed_minus_cost_regret": _mean(routed_regret - cost_regret),
            "unreleased_but_improvable_count": int(missed.sum()),
            "unreleased_but_improvable_rate": _rate(missed),
            "unreleased_but_improvable_rate_among_effective": _safe_ratio(int(missed.sum()), int(effective_improvement.sum())),
            "unreleased_but_improvable_roots": [row["root"] for row, value in zip(parsed, missed) if value],
            "claim_boundary": "The release flag is the report's already-frozen router decision. These are descriptive post-execution operating characteristics, not a new authorization or retrospective threshold search.",
        },
    }


def audit(report: dict[str, Any]) -> dict[str, Any]:
    """Audit calibration and outcome rows without consulting outcome for policy choice."""

    calibration = report.get("calibration")
    outcome = report.get("outcome")
    if not isinstance(calibration, dict) or not isinstance(outcome, dict):
        raise ValueError("report requires calibration and outcome objects")
    control_contract = report.get("control_contract")
    declared_opportunity_gate: bool | None = None
    if isinstance(control_contract, dict) and "opportunity_gated" in control_contract:
        value = control_contract["opportunity_gated"]
        if not isinstance(value, bool):
            raise ValueError("control_contract.opportunity_gated must be boolean")
        declared_opportunity_gate = value
    return {
        "schema": SCHEMA,
        "status": "completed_post_execution_mechanism_audit",
        "claim_boundary": "A post-execution decomposition of a completed V5 planning-shaped repairability report. It is not a policy selector, prospective repairability result, PlanWAM validation, GDD validation, or deployment evidence.",
        "calibration": _partition(
            calibration.get("per_root"), name="calibration",
            opportunity_gated=declared_opportunity_gate,
        ),
        "outcome": _partition(
            outcome.get("per_root"), name="outcome",
            opportunity_gated=declared_opportunity_gate,
        ),
        "interpretation_contract": {
            "candidate_disagreement_is_necessary_not_sufficient_for_arm_gain": True,
            "all_metrics_are_stratified_before_any_future_protocol_change": True,
            "outcome_metrics_must_not_be_used_to_choose_a_new_router_threshold": True,
            "fresh_root_disjoint_panel_required_for_prospective_claim": True,
        },
    }


def run(*, report_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not report_path.is_file():
        raise FileNotFoundError(f"report does not exist: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    result = {
        **audit(report),
        "provenance": {"report": str(report_path.resolve()), "report_sha256": _sha256(report_path)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(report_path=args.report, output=args.output)
    print(json.dumps({
        "status": result["status"],
        "outcome_disagreement": result["outcome"]["candidate_choice"]["disagreement_count"],
        "outcome_effective_improvement": result["outcome"]["planning_shaped_effect"]["strict_improvement_count_when_disagree"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
