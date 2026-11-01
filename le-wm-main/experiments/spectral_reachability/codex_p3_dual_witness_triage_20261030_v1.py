"""Pure strict-R witness triage for a future, separately sealed P3 consumer.

No model, physical executor, or outcome-dependent bank is provided. A valid
classification is not source authentication or an authorization to dispatch.
"""
import math

from experiments.spectral_reachability import p3_decision_dual_branch_20261030_codex_v1 as dual

need = dual.need
NAME = "codex_p3_dual_witness_triage_20261030_v1"


def scalar(value):
    need(type(value) in (float, int) and math.isfinite(value), "finite scalar required")
    return value


def audit_panel(panel, horizon):
    """Recompute the original strict inequalities from all nine saved rows."""
    need(type(panel) is dict and horizon in (1, 3) and
         panel["bank"]["horizon"] == horizon, "panel horizon")
    control, rows = panel["control"], panel["rows"]
    need(control["index"] == "zero" and control["status"] == "complete" and
         type(rows) is list and len(rows) == 8 and
         [r["index"] for r in rows] == list(range(8)) and
         all(r["status"] == "complete" for r in rows), "complete zero plus eight")
    immediate, reach = scalar(control["immediate"]), scalar(control["reach"])
    values = []
    for row in rows:
        a, b = scalar(row["immediate"]), scalar(row["reach"])
        qualified = a > immediate and b < reach
        need(type(row["eligible"]) is bool and row["eligible"] == qualified, "strict witness mismatch")
        values.append(dict(index=row["index"], immediate_gt_zero=a > immediate,
                           reach_lt_zero=b < reach, eligible=qualified))
    eligible = [r["index"] for r in values if r["eligible"]]
    need(panel["eligible_indices"] == eligible and
         panel["status"] == ("witness_found" if eligible else "no_witness"), "panel status/census")
    diagnostic = min(eligible, key=lambda i: (rows[i]["reach"], i)) if eligible else None
    need(panel["diagnostic_reach_winner"] == diagnostic, "diagnostic winner")
    winner = min(eligible, key=lambda i: (rows[i]["immediate"], i)) if eligible else None
    return dict(horizon=horizon, status=panel["status"], eligible_indices=eligible,
                selection_index=winner, rows=values)


def classify(decision):
    """Classify only the complete, original no-witness terminal; never dispatch."""
    need(decision["schema"] == dual.NAME + ".decision" and decision["contract"] == dual.contract()
         and decision["branch"] == "reconfiguration", "original dual R contract")
    dual.validate_decision(decision["decision"])
    need(decision["decision"]["extension_type"] == "reconfiguration", "original branch")
    need(decision["status"] == "failed" and
         decision["error"] == "ValueError: R group has no witness; both panels required" and
         decision["selected_actions"] is None and decision["selected_horizon"] is None,
         "original failure terminal")
    panels = decision["panels"]
    need(type(panels) is list and len(panels) == 2, "both complete horizon panels required")
    audit = [audit_panel(panel, h) for panel, h in zip(panels, (1, 3), strict=True)]
    need(any(not p["eligible_indices"] for p in audit), "failure requires absent witness")
    dual.core.audit_ledger(decision["resource_receipt"])
    return dict(schema=NAME + ".classification", classification="strict_no_witness",
                stage=decision["stage"], panels=audit, dispatch_authorized=False,
                selected_actions=None, source_authenticated=False, physical_certified=False,
                scientific_pass=False, original_run_status="failed")


def future_contract():
    """Specification only; not a claim of preregistration or a launcher."""
    return dict(schema=NAME + ".future_contract", original_contract=dual.contract(),
                no_witness="retain fixed roster; no packet, retry, fallback or substitution",
                exception="separate from no_witness; retain attempted work",
                required_before_future_run=["new protocol and complete source authority sealed before new outcomes",
                    "fresh independent review and explicit host authorization",
                    "strict stopped-service saved consumer and externally pinned terminal",
                    "separately specified fixed denominator and missing-endpoint analysis"],
                planning_or_physical_authorized=False, scientific_pass=False)
