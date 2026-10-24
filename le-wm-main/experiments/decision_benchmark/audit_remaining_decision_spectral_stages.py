"""Close P4--P7 by evaluating their upstream eligibility, not by bypassing it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _p3_replication_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one P3 report is required")
    seen_seeds: set[int] = set()
    rows: list[dict[str, Any]] = []
    for report in reports:
        protocol, summary = report.get("protocol"), report.get("summary")
        if not isinstance(protocol, dict) or not isinstance(summary, dict):
            raise ValueError("P3 report is missing protocol or summary")
        seed = protocol.get("seed")
        if seed is not None:
            if not isinstance(seed, int) or seed in seen_seeds:
                raise ValueError("P3 replication seeds must be distinct integers when declared")
            seen_seeds.add(seed)
        values = {name: float(summary[name]["mean_realized_regret"]) for name in (
            "spectral_rank1", "random_rank1", "pca_axis_rank1", "ordinary_full_effect",
        )}
        rank_matched_passed = (
            values["spectral_rank1"] < values["random_rank1"]
            and values["spectral_rank1"] < values["pca_axis_rank1"]
        )
        rows.append({
            "seed": seed,
            "passes_rank_matched_controls": rank_matched_passed,
            "full_effect_capacity_reference_beats_rank1": values["ordinary_full_effect"] < values["spectral_rank1"],
            **values,
        })
    return {
        "replication_count": len(rows),
        "all_replicates_pass_rank_matched_controls": all(row["passes_rank_matched_controls"] for row in rows),
        "full_effect_is_capacity_reference_not_primary_gate": True,
        "replicates": rows,
    }


def _p3v2_sweep_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Validate that the corrected P3 sweep varies only the declared metric."""

    if report.get("schema") != "spectral_repair_synthetic_p3_v2_task_metric_sweep_cuda.v2":
        raise ValueError("P3v2 task-metric sweep has an unexpected schema")
    if report.get("status") != "completed_task_metric_mechanism_sweep" or report.get("common_random_numbers") is not True:
        raise ValueError("P3v2 sweep did not complete with common random numbers")
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("P3v2 sweep has no records")
    seeds = {row.get("seed") for row in records if isinstance(row, dict)}
    if len(seeds) != 1 or not all(isinstance(row, dict) for row in records):
        raise ValueError("P3v2 sweep confounds task weight with synthetic seeds")
    passed_weights = [float(row["task_output_0_weight"]) for row in records if row.get("status") == "passed_task_weighted_rank_matched_calibration"]
    return {
        "status": "conditional_task_critical_mode_calibration_pass" if passed_weights else "no_task_critical_mode_calibration_pass",
        "common_random_numbers": True,
        "seed": next(iter(seeds)),
        "passed_task_output_0_weights": passed_weights,
        "all_task_output_0_weights": [float(row["task_output_0_weight"]) for row in records],
        "claim_boundary": "A pass at declared high task weights validates the controlled mode-selection mechanism only; it does not authorize a real repair or establish a universal effect.",
    }


def _p1_endpoint_readout_binding(audit: dict[str, Any]) -> dict[str, Any]:
    """Require the P1 outcome association to use a verified endpoint policy."""

    outcome = audit.get("splits", {}).get("outcome")
    rows = outcome.get("per_root") if isinstance(outcome, dict) else None
    if not isinstance(rows, dict) or not rows:
        raise ValueError("P1 audit lacks outcome-root endpoint readout bindings")
    bindings = []
    for root, row in rows.items():
        binding = row.get("endpoint_readout_binding") if isinstance(row, dict) else None
        if not isinstance(binding, dict):
            raise ValueError(f"P1 audit root {root} lacks endpoint readout binding")
        if binding.get("endpoint_candidate_is_deterministic_argmin") is not True:
            raise ValueError(f"P1 audit root {root} does not bind endpoint choice to its score")
        if binding.get("realized_regret_recomputed_from_same_candidate_ordering") is not True:
            raise ValueError(f"P1 audit root {root} does not bind realized regret to the sealed candidate ordering")
        error = binding.get("maximum_absolute_cost_error")
        if not isinstance(error, (int, float)) or not np.isfinite(error) or float(error) > 1.0e-10:
            raise ValueError(f"P1 audit root {root} has an invalid endpoint readout cost error")
        bindings.append(float(error))
    return {
        "status": "verified_endpoint_readout_binding",
        "root_count": len(bindings),
        "maximum_absolute_cost_error": max(bindings),
        "contract": "endpoint candidate is the deterministic Euclidean terminal-to-goal argmin and realized regret uses that same candidate ordering",
    }


def _p1_execution_identity(audit: dict[str, Any]) -> dict[str, Any]:
    """Require an independent replay before treating P1 as an outcome audit."""

    if audit.get("schema") != "pointmaze_endpoint_execution_identity_audit.v1":
        raise ValueError("P1 identity audit has an unexpected schema")
    if audit.get("status") != "passed_independent_outcome_identity_replay":
        raise ValueError("P1 identity audit did not pass independent outcome replay")
    roots, candidates = audit.get("outcome_root_count"), audit.get("outcome_candidate_count")
    if not isinstance(roots, int) or roots < 1 or not isinstance(candidates, int) or candidates < roots:
        raise ValueError("P1 identity audit has invalid root/candidate coverage")
    errors = (audit.get("maximum_terminal_absolute_error"), audit.get("maximum_cost_absolute_error"))
    if any(not isinstance(value, (int, float)) or not np.isfinite(value) or float(value) > 1.0e-10 for value in errors):
        raise ValueError("P1 identity audit has non-zero or invalid replay error")
    return {
        "status": "verified_independent_outcome_identity_replay",
        "root_count": roots,
        "candidate_count": candidates,
        "maximum_terminal_absolute_error": float(errors[0]),
        "maximum_cost_absolute_error": float(errors[1]),
    }


def _p1_prime_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Record a post-execution routing audit without promoting it to P1 pass."""

    if report.get("schema") != "pointmaze_v6_prospective_routing_post_execution_audit.v1":
        raise ValueError("P1-prime audit has an unexpected schema")
    if report.get("status") != "completed_post_execution_prospective_routing_audit":
        raise ValueError("P1-prime audit did not complete")
    decision = report.get("decision")
    certification = report.get("certification")
    if not isinstance(decision, dict) or not isinstance(certification, dict):
        raise ValueError("P1-prime audit lacks its decision/certification record")
    warranted = bool(decision.get("fresh_panel_route_warranted"))
    return {
        "status": "fresh_panel_warranted" if warranted else "no_go_uncertified_pre_outcome_route",
        "fresh_panel_route_warranted": warranted,
        "calibration_authorized": bool(certification.get("would_authorize_on_calibration")),
        "reason": (
            "A route transferred through its independent calibration certificate; a new prospective panel is eligible."
            if warranted else
            "No pre-outcome route met the independent calibration authorization; it cannot be promoted to a repair policy."
        ),
    }


def _real_repair_preflight_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Preserve a static candidate-bank no-go separately from model efficacy."""

    if report.get("schema") != "pointmaze_endpoint_repairability_router_v10_static_preflight.v1":
        raise ValueError("real-repair preflight has an unexpected schema")
    status = report.get("status")
    if status not in {"passed_static_preflight", "rejected_static_preflight"}:
        raise ValueError("real-repair preflight has an invalid status")
    checks = report.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("real-repair preflight lacks checks")
    failed = sorted(str(name) for name, value in checks.items() if value is not True)
    return {
        "status": "eligible_for_cuda_execution" if status == "passed_static_preflight" else "blocked_pre_fit_candidate_bank",
        "static_preflight_status": status,
        "failed_checks": failed,
        "claim_boundary": "A rejected static preflight is not a learned-model failure or a negative theory result; it prevents an unidentifiable candidate bank from entering CUDA fitting.",
    }


def audit(*, v6r2_report: Path, p1_audit: Path, p1_identity_audit: Path, p1_prime_audit: Path, p2_report: Path, p3_report: Path,
          p3v2_sweep: Path, real_repair_preflight: Path, output: Path, p3_replication_reports: tuple[Path, ...] = ()) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    v6r2, p1, p1_identity, p1_prime, p2, p3v2, real_repair = (
        _load(path) for path in (v6r2_report, p1_audit, p1_identity_audit, p1_prime_audit, p2_report, p3v2_sweep, real_repair_preflight)
    )
    p3_replications = _p3_replication_summary([
        _load(path) for path in (p3_report, *p3_replication_reports)
    ])
    p1_outcome = p1["splits"]["outcome"].get("primary_metric_association", p1["splits"]["outcome"]["spectral_mismatch_vs_endpoint_regret"])
    p1_binding = _p1_endpoint_readout_binding(p1)
    p1_identity_summary = _p1_execution_identity(p1_identity)
    p1_pass = bool(p1_outcome.get("spearman", 0.0) > 0.0 and p1_outcome.get("two_sided_permutation_p", 1.0) < 0.05)
    p2_pass = p2.get("status") == "passed_synthetic_formula_calibration"
    p3_legacy_rank_matched_pass = bool(p3_replications["all_replicates_pass_rank_matched_controls"])
    p1_prime_summary = _p1_prime_summary(p1_prime)
    p3v2_summary = _p3v2_sweep_summary(p3v2)
    real_repair_summary = _real_repair_preflight_summary(real_repair)
    p3_synthetic_mechanism_pass = p3v2_summary["status"] == "conditional_task_critical_mode_calibration_pass"
    p4_eligible = p1_pass and p3_synthetic_mechanism_pass and real_repair_summary["status"] == "eligible_for_cuda_execution"
    result: dict[str, Any] = {
        "schema": "decision_spectral_remaining_stage_gate_audit.v2",
        "status": "completed_all_stage_eligibility_audit",
        "stages": {
            "P1_decision_spectral_prediction": {
                "status": "passed" if p1_pass else "no_go_held_out_association",
                "primary_metric": p1.get("protocol", {}).get("primary_metric", "spectral_mismatch_norm"),
                "outcome_spearman": p1_outcome.get("spearman"),
                "two_sided_permutation_p": p1_outcome.get("two_sided_permutation_p"),
                "endpoint_readout_binding": p1_binding,
                "outcome_execution_identity": p1_identity_summary,
            },
            "P1_prime_pre_outcome_repairability_route": p1_prime_summary,
            "P2_synthetic_mode_selection": {"status": "passed_formula_calibration" if p2_pass else "failed", "scope": "CUDA controlled mechanism only"},
            "P3_rank_matched_spectral_repair": {
                "status": "conditional_synthetic_mechanism_pass" if p3_synthetic_mechanism_pass else "no_go_synthetic_mechanism",
                "replication_summary": p3_replications,
                "legacy_rank_matched_controls_pass": p3_legacy_rank_matched_pass,
                "task_weighted_common_random_numbers_sweep": p3v2_summary,
                "reason": "A rank-one method may only be primarily compared to rank-one controls. The unrestricted full-effect arm is a capacity reference; real repair still requires P1 and a passed fresh candidate-bank preflight.",
                "execution_grounded_endpoint_repair_preflight": real_repair_summary,
            },
            "P4_recalibrated_HTSC": {
                "status": "blocked_upstream" if not p4_eligible else "eligible_not_yet_run",
                "current_v6r2_policy": v6r2["calibration"]["policy"]["deployment_policy"],
                "reason": "there is no validated repaired student to recalibrate" if not p4_eligible else "requires a frozen validated repaired student",
            },
            "P5_RGB_only_PointMaze_Reacher_MPC": {
                "status": "blocked_upstream" if not p4_eligible else "eligible_not_yet_run",
                "reason": "a renderer-backed fixed-bank endpoint panel is not RGB-only closed-loop MPC, and no validated repair exists" if not p4_eligible else "requires P4 pass and new root-disjoint online episodes",
            },
            "P6_PushT_contact_cross_model": {
                "status": "blocked_upstream" if not p4_eligible else "eligible_not_yet_run",
                "reason": "cannot transfer an unvalidated repair mechanism to contact dynamics" if not p4_eligible else "requires P5 mechanism pass plus a stable visual PushT baseline",
            },
            "P7_external_PlanWAM_style_transfer": {
                "status": "blocked_upstream" if not p4_eligible else "eligible_not_yet_run",
                "reason": "external transfer cannot establish a mechanism that P1/P3 did not validate" if not p4_eligible else "requires P6 pass and a declared external simulator protocol",
            },
        },
        "claim_boundary": "This is a completed stage-gate audit, not evidence that blocked stages ran or that their outcome is negative. It prevents invalid downstream commands from laundering an upstream no-go into a repair, certificate, MPC, contact, or transfer claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6r2-report", type=Path, required=True); parser.add_argument("--p1-audit", type=Path, required=True); parser.add_argument("--p1-identity-audit", type=Path, required=True)
    parser.add_argument("--p1-prime-audit", type=Path, required=True)
    parser.add_argument("--p2-report", type=Path, required=True); parser.add_argument("--p3-report", type=Path, required=True)
    parser.add_argument("--p3v2-sweep", type=Path, required=True)
    parser.add_argument("--real-repair-preflight", type=Path, required=True)
    parser.add_argument("--p3-replication-report", dest="p3_replication_reports", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    args["p3_replication_reports"] = tuple(args["p3_replication_reports"])
    result = audit(**args)
    print(json.dumps({"status": result["status"], "stages": result["stages"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
