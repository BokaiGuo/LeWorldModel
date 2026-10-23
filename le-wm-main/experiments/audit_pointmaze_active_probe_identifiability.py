"""Audit whether an RGB active probe could identify a local action chart.

This is a mechanism audit, not an efficacy evaluator.  In particular, a
trusted residual model fitted from four task-safe pulses can still be severely
underdetermined when its feature basis contains action/velocity interactions.
The audit reconstructs the *decision-time* probe geometry from immutable raw
traces and separates five explanations for a missing local override:

* no information action was triggered;
* probe actions did not span an affine action chart;
* the legacy 12-feature residual was algebraically underdetermined;
* successor verification rejected the observed chart; or
* a trusted chart had no supported candidate / was not released by its final
  visual gate.

It deliberately consumes neither simulator state nor held-out task outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_pointmaze_htsc_active_probe_strict import ACTIVE_METHOD
from experiments.run_live_pointmaze_planning import (
    OnlineLocalEffectResidual,
    PriorAnchoredResidualJacobianModel,
)


SCHEMA = "pointmaze.active_probe_identifiability_audit.v1"
AFFINE_ACTIVE_METHODS = frozenset({
    "effect_adapter_rgb_online_affine_verified_local",
    "effect_adapter_rgb_online_affine_candidate_anchored_verified_local",
})
PRIOR_ANCHORED_RESIDUAL_METHOD = (
    "effect_adapter_rgb_online_prior_anchored_candidate_residual_verified_local"
)
PRIOR_ANCHORED_CONTRAST_METHOD = (
    "effect_adapter_rgb_online_prior_anchored_candidate_residual_contrast_verified_local"
)
CANDIDATE_ANCHORED_PROBE_DESIGN = "candidate_anchored_task_safe_affine_d_optimal"
PRIOR_ANCHORED_PROBE_DESIGN = "candidate_anchored_task_safe_prior_residual_rank2"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _rank_and_condition(matrix: np.ndarray) -> tuple[int, float]:
    if matrix.ndim != 2 or not len(matrix):
        return 0, float("inf")
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    if not len(singular) or rank < min(matrix.shape) or singular[-1] <= 1.0e-12:
        return rank, float("inf")
    return rank, float(singular[0] / singular[-1])


def _root_diagnostic(row: dict[str, Any]) -> dict[str, Any]:
    events = [
        event for event in row.get("active_probe_trace", [])
        if isinstance(event, dict) and "probe_index" in event and "error" not in event
    ]
    gates = [
        item.get("action_transfer_gate", {}) for item in row.get("trace", [])
        if isinstance(item, dict) and isinstance(item.get("action_transfer_gate"), dict)
    ]
    preplanning = [
        event for event in row.get("active_probe_trace", [])
        if isinstance(event, dict) and bool(event.get("preplanning_affine_repairability", False))
    ]
    if len(preplanning) > 1:
        raise ValueError(f"root {row.get('episode_id')}: duplicate preplanning affine diagnostic")
    preplanning_diagnostic = preplanning[0] if preplanning else None
    if not events:
        return {
            "episode_id": int(row["episode_id"]),
            "probe_count": 0,
            "failure_mode": "probe_not_triggered_or_early_rejected",
            "local_override_count": 0,
            "htsc_nonrelease_count": int(sum(
                not bool(gate.get("effect_policy_released", True)) for gate in gates
            )),
        }

    actions = np.asarray([event["action"] for event in events], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 2 or not np.isfinite(actions).all():
        raise ValueError(f"root {row.get('episode_id')}: invalid probe action trace")
    affine = np.concatenate([np.ones((len(actions), 1)), actions], axis=1)
    prior_residual = actions
    residual = OnlineLocalEffectResidual._features(actions)
    affine_rank, affine_condition = _rank_and_condition(affine)
    residual_rank, residual_condition = _rank_and_condition(residual)
    residual_model = OnlineLocalEffectResidual(actions=[action.copy() for action in actions])
    prior_model = PriorAnchoredResidualJacobianModel(actions=[action.copy() for action in actions])
    candidates = np.asarray(row.get("policy_contract", {}).get("candidate_actions", []), dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1:] != (2,) or not len(candidates):
        raise ValueError(f"root {row.get('episode_id')}: missing fixed candidate actions")
    prior_anchored_arm = str(row.get("method")) in {
        PRIOR_ANCHORED_RESIDUAL_METHOD, PRIOR_ANCHORED_CONTRAST_METHOD,
    }
    supported = (
        prior_model.action_support_mask(candidates)
        if prior_anchored_arm else residual_model.action_support_mask(candidates)
    )
    candidate_anchored_events = [
        event for event in events
        if event.get("probe_design") in {
            CANDIDATE_ANCHORED_PROBE_DESIGN, PRIOR_ANCHORED_PROBE_DESIGN,
        }
    ]
    anchor_membership = []
    for event in candidate_anchored_events:
        candidate_index = event.get("probe_candidate_index")
        valid_index = isinstance(candidate_index, int) and 0 <= candidate_index < len(candidates)
        anchor_membership.append(bool(
            valid_index and np.allclose(
                np.asarray(event.get("action"), dtype=np.float64),
                candidates[candidate_index], atol=1.0e-10, rtol=0.0,
            )
        ))
    verification_errors = [
        float(event["verification_error"]) for event in events
        if event.get("verification_error") is not None
    ]
    trusted = bool(events[-1].get("residual_trusted", False))
    nonrelease = [gate for gate in gates if not bool(gate.get("effect_policy_released", True))]
    overrides = [
        gate for gate in gates
        if str(gate.get("fallback_reason", "")).endswith("local_verified_override")
        or str(gate.get("fallback_reason", "")).endswith("affine_local_override")
        or str(gate.get("fallback_reason", "")).endswith("residual_local_override")
    ]
    local_support_seen = any(
        bool(gate.get("local_action_support_passed", False))
        or int(gate.get("local_affine_supported_candidate_count", 0)) > 0
        for gate in nonrelease
    )
    affine_arm = str(row.get("method")) in AFFINE_ACTIVE_METHODS
    prior_rank, prior_condition = _rank_and_condition(prior_residual)
    if prior_anchored_arm and prior_rank < prior_residual.shape[1]:
        failure_mode = "prior_anchored_residual_chart_not_identifiable"
    elif affine_arm and affine_rank < affine.shape[1]:
        failure_mode = "affine_chart_not_identifiable"
    elif affine_arm and preplanning_diagnostic is not None and not bool(
        preplanning_diagnostic.get("local_chart_loo_reliable", False)
    ):
        failure_mode = "affine_chart_crossfit_unreliable"
    elif affine_arm and not np.any(supported):
        failure_mode = "no_candidate_in_affine_probe_hull"
    elif not affine_arm and not prior_anchored_arm and residual_rank < residual.shape[1]:
        failure_mode = "legacy_residual_basis_underdetermined"
    elif not trusted:
        failure_mode = "successor_verification_not_trusted"
    elif not nonrelease:
        failure_mode = "no_htsc_nonrelease_after_probe"
    elif not local_support_seen:
        failure_mode = "no_selected_candidate_in_local_probe_hull"
    elif not overrides:
        failure_mode = "post_probe_release_gate_rejected"
    else:
        failure_mode = "local_override_exercised"
    return {
        "episode_id": int(row["episode_id"]),
        "probe_count": int(len(events)),
        "probe_actions": actions.tolist(),
        "affine_design_rank": affine_rank,
        "affine_feature_count": int(affine.shape[1]),
        "affine_design_condition_number": affine_condition,
        "prior_anchored_residual_design_rank": prior_rank,
        "prior_anchored_residual_feature_count": int(prior_residual.shape[1]),
        "prior_anchored_residual_design_condition_number": prior_condition,
        "local_chart_basis": (
            "calibration_prior_zero_action_plus_observed_rgb_residual_jacobian"
            if prior_anchored_arm else "observed_rgb_affine_bias_plus_action_jacobian"
        ),
        "legacy_residual_design_rank": residual_rank,
        "legacy_residual_feature_count": int(residual.shape[1]),
        "legacy_residual_design_condition_number": residual_condition,
        "legacy_residual_is_algebraically_identifiable": bool(residual_rank == residual.shape[1]),
        "candidate_bank_size": int(len(candidates)),
        "candidate_hull_coverage_count": int(np.sum(supported)),
        "candidate_hull_coverage_fraction": float(np.mean(supported)),
        "candidate_anchored_probe_event_count": int(len(candidate_anchored_events)),
        "candidate_anchored_probe_membership_passed": (
            None if not candidate_anchored_events else bool(all(anchor_membership))
        ),
        "residual_trusted_after_last_probe": trusted,
        "probe_verification_error_mean": (
            None if not verification_errors else float(np.mean(verification_errors))
        ),
        "probe_verification_error_max": (
            None if not verification_errors else float(np.max(verification_errors))
        ),
        "htsc_nonrelease_count": int(len(nonrelease)),
        "local_support_seen_after_probe": bool(local_support_seen),
        "local_override_count": int(len(overrides)),
        "preplanning_affine_repairability": preplanning_diagnostic,
        "failure_mode": failure_mode,
    }


def audit(report: dict[str, Any], active_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not active_rows:
        raise ValueError("active raw report has no rows")
    diagnostics = [_root_diagnostic(row) for row in active_rows]
    modes: dict[str, int] = {}
    for diagnostic in diagnostics:
        mode = str(diagnostic["failure_mode"])
        modes[mode] = modes.get(mode, 0) + 1
    probed = [row for row in diagnostics if int(row["probe_count"]) > 0]
    return {
        "schema": SCHEMA,
        "status": "completed_mechanism_diagnostic",
        "active_method": active_rows[0].get("method", ACTIVE_METHOD),
        "root_count": int(len(diagnostics)),
        "probed_root_count": int(len(probed)),
        "failure_mode_counts": dict(sorted(modes.items())),
        "summary": {
            "mean_affine_candidate_hull_coverage": (
                None if not probed else float(np.mean([
                    row["candidate_hull_coverage_fraction"] for row in probed
                ]))
            ),
            "legacy_residual_underdetermined_probed_roots": int(sum(
                row["failure_mode"] == "legacy_residual_basis_underdetermined"
                for row in probed
            )),
            "prior_anchored_rank_two_identifiable_probed_roots": int(sum(
                row["local_chart_basis"]
                == "calibration_prior_zero_action_plus_observed_rgb_residual_jacobian"
                and int(row["prior_anchored_residual_design_rank"])
                == int(row["prior_anchored_residual_feature_count"])
                for row in probed
            )),
            "local_override_exercised_roots": int(sum(
                int(row["local_override_count"]) > 0 for row in diagnostics
            )),
            "candidate_anchored_probe_event_count": int(sum(
                int(row.get("candidate_anchored_probe_event_count", 0)) for row in diagnostics
            )),
            "candidate_anchored_probe_membership_failed_roots": int(sum(
                row.get("candidate_anchored_probe_membership_passed") is False
                for row in diagnostics
            )),
        },
        "per_root": diagnostics,
        "claim_boundary": (
            "This audit reconstructs information-action identifiability and release-path usage from raw RGB traces. "
            "It neither validates task benefit nor retroactively authorizes a calibration-rejected controller."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-method", default=ACTIVE_METHOD)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    report = _load(args.report)
    reports = report.get("artifacts", {}).get("held_out_arm_reports", {})
    raw_path = reports.get(args.active_method)
    if raw_path is None:
        raise ValueError(f"aggregate report lacks active arm {args.active_method!r}")
    raw = _load(Path(str(raw_path)))
    active_rows = [
        row for row in raw.get("rows", [])
        if isinstance(row, dict) and row.get("method") == args.active_method
    ]
    result = audit(report, active_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "summary": result["summary"],
        "failure_mode_counts": result["failure_mode_counts"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
