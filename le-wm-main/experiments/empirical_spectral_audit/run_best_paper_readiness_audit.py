"""Audit the three Best-Paper requirements against immutable artifacts.

The tool is intentionally conservative.  It provides a reusable finite-horizon
``G_h`` report and checkpoint ranking, but it grants learned-action permission
only when independent action-identifiability, planner-reachable transition,
readout calibration, and real/predicted successor gates are present.  It also
emits an explicit equal-spectrum counterexample showing why the transfer map,
not a scalar spectral diagnostic, is the decision-relevant object.

This script audits evidence; it does not tune a checkpoint on held-out
planning outcomes and does not turn a synthetic counterexample into learned
world-model evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .run_matched_learned_transfer_benchmark import _load
except ImportError:  # direct script execution
    from run_matched_learned_transfer_benchmark import _load  # type: ignore[no-redef]

try:
    from ..theory_spectral_transfer.finite_horizon_transfer import (
        finite_horizon_response,
        spectrum_alignment_commutator_counterexample,
        transfer_bound_tightness_example,
    )
except ImportError:  # direct script execution from the project root
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.theory_spectral_transfer.finite_horizon_transfer import (  # type: ignore[no-redef]
        finite_horizon_response,
        spectrum_alignment_commutator_counterexample,
        transfer_bound_tightness_example,
    )


HORIZONS = (1, 2, 4, 8, 16)
EXTERNAL_GAIN_FIELDS = (
    "action_ranking_inversion",
    "multistep_task_error",
    "contact_retention",
    "angle_closure",
    "planning_regret",
)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _operator(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    diagnostics = record.get("diagnostics", {})
    operator = diagnostics.get("transfer_operator")
    if not isinstance(operator, dict):
        return None
    try:
        A = np.asarray(operator["A"], dtype=np.float64)
        B = np.asarray(operator["B"], dtype=np.float64)
        D = np.asarray(operator["D"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if A.ndim != 2 or B.ndim != 2 or D.ndim != 2:
        return None
    if A.shape[0] != A.shape[1] or B.shape[0] != A.shape[0] or D.shape[1] != A.shape[0]:
        return None
    if not np.isfinite(A).all() or not np.isfinite(B).all() or not np.isfinite(D).all():
        return None
    return A, B, D


def _markov_report(record: dict[str, Any], horizons: tuple[int, ...]) -> dict[str, Any]:
    operator = _operator(record)
    if operator is None:
        return {"status": "missing_transfer_operator"}
    A, B, D = operator
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    output_norm = float(np.linalg.norm(D, 2))
    maps: dict[str, Any] = {}
    for horizon in horizons:
        response = finite_horizon_response(A, B, D, horizon)
        finite_gains = [float(np.linalg.norm(D @ np.linalg.matrix_power(A, lag), 2)) for lag in range(horizon)]
        spectral_gains = [output_norm * rho**lag for lag in range(horizon)]
        maps[str(horizon)] = {
            "G_h": response.tolist(),
            "G_h_norm": float(np.linalg.norm(response, 2)),
            "finite_horizon_task_gain": max(finite_gains, default=0.0),
            "spectral_only_task_gain": max(spectral_gains, default=0.0),
            "transient_excess": max(
                [finite - spectral for finite, spectral in zip(finite_gains, spectral_gains)] or [0.0]
            ),
        }
    return {
        "status": "available",
        "spectral_radius": rho,
        "output_norm": output_norm,
        "horizons": maps,
        "risk_proxy_definition": "G_h norm plus finite-horizon task gain and excess over spectral-radius-only gain",
    }


def _permission(record: dict[str, Any]) -> dict[str, Any]:
    diagnostics = record.get("diagnostics", {})
    operator = diagnostics.get("transfer_operator", {})
    envelope = operator.get("risk_envelope", {}) if isinstance(operator, dict) else {}
    action_text = str(envelope.get("action_effect", ""))
    off_text = str(envelope.get("off_manifold", ""))
    readout_text = str(envelope.get("readout", ""))
    gates = {
        "action_identifiability": bool(envelope.get("action_identifiability_passed", False))
        and "unidentified" not in action_text.lower(),
        "planner_reachable_off_manifold": bool(envelope.get("off_manifold_passed", False))
        and "unidentified" not in off_text.lower(),
        "independent_readout": bool(envelope.get("readout_passed", False))
        and "independent" not in readout_text.lower(),
        "real_predicted_transition": bool(envelope.get("transition_consistency_passed", False)),
        "certified_bound": bool(envelope.get("bound_allowed_for_control", False)),
    }
    warnings = []
    if not gates["action_identifiability"]:
        warnings.append("action_identifiability_missing")
    if not gates["planner_reachable_off_manifold"]:
        warnings.append("planner_reachable_transition_missing")
    if not gates["independent_readout"]:
        warnings.append("independent_task_readout_missing")
    if not gates["real_predicted_transition"]:
        warnings.append("real_predicted_successor_consistency_missing")
    if not gates["certified_bound"]:
        warnings.append("finite_horizon_bound_not_released_for_control")
    return {
        "status": "permitted" if all(gates.values()) else "rejected_fail_closed",
        "gates": gates,
        "warnings": warnings,
        "risk_envelope_status": envelope.get("status", "missing"),
    }


def _load_records(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records = []
    errors = []
    for path in sorted(root.rglob("checkpoint_record.json")):
        try:
            records.append(_load(path))
        except (ValueError, KeyError, TypeError) as exc:
            errors.append({"path": str(path), "reason": str(exc)})
    return records, errors


def _external_gain_gate(results_root: Path) -> dict[str, Any]:
    """Accept only an explicit independent-gain schema, never episode guessing."""
    candidates = []
    for path in sorted(results_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        gate = payload.get("external_independent_gain")
        if isinstance(gate, dict):
            candidates.append({"path": str(path), "gate": gate})
    valid = []
    invalid_reasons: list[dict[str, Any]] = []
    for item in candidates:
        gate = item["gate"]
        schema_ok = gate.get("schema") == "external_independent_gain.v1"
        metrics = gate.get("metrics", {})
        field_reasons: list[str] = []
        for field in EXTERNAL_GAIN_FIELDS:
            value = metrics.get(field) if isinstance(metrics, dict) else None
            if not isinstance(value, dict):
                field_reasons.append(f"{field}:missing_metric_object")
                continue
            if value.get("status") != "available":
                field_reasons.append(f"{field}:status={value.get('status', 'missing')}")
                continue
            if not _finite(value.get("estimate")):
                field_reasons.append(f"{field}:nonfinite_estimate")
            interval = value.get("ci95")
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or not all(_finite(bound) for bound in interval)
            ):
                field_reasons.append(f"{field}:invalid_ci95")
        fields_ok = not field_reasons
        protocol = gate.get("protocol", {})
        no_privileged = protocol.get("privileged_state_used") is False
        no_test_selection = protocol.get("test_planning_used_for_selection") is False
        split_ok = protocol.get("train_calibration_test_root_disjoint") is True
        heldout_ok = protocol.get("independent_heldout_consequence") is True
        if schema_ok and fields_ok and no_privileged and no_test_selection and split_ok and heldout_ok:
            valid.append(item)
        else:
            invalid_reasons.append(
                {
                    "path": item["path"],
                    "reasons": [
                        *([] if schema_ok else ["schema_mismatch"]),
                        *field_reasons,
                        *([] if no_privileged else ["privileged_state_used_not_false"]),
                        *([] if no_test_selection else ["test_planning_used_for_selection_not_false"]),
                        *([] if split_ok else ["train_calibration_test_root_disjoint_not_true"]),
                        *([] if heldout_ok else ["independent_heldout_consequence_not_true"]),
                    ],
                }
            )
    return {
        "status": "available" if valid else "blocked_missing_independent_gain_schema",
        "candidate_artifact_count": len(candidates),
        "valid_artifact_count": len(valid),
        "required_fields": list(EXTERNAL_GAIN_FIELDS),
        "valid_artifacts": valid,
        "invalid_artifacts": invalid_reasons,
        "claim_boundary": "episode-level planning logs and privileged-state runs are not accepted as the independent gain gate",
    }


def audit(record_root: Path, results_root: Path, horizons: tuple[int, ...] = HORIZONS) -> dict[str, Any]:
    records, errors = _load_records(record_root)
    reports = []
    for record in records:
        markov = _markov_report(record, horizons)
        reports.append({
            "architecture": record.get("_architecture", record.get("architecture", "unknown")),
            "environment": record.get("_environment", "unknown"),
            "seed": record.get("_seed", "unknown"),
            "training_stage": record.get("_stage"),
            "checkpoint": record.get("checkpoint"),
            "finite_horizon_transfer": markov,
            "permission": _permission(record),
        })
    available = [item for item in reports if item["finite_horizon_transfer"]["status"] == "available"]
    rankings = {}
    for horizon in horizons:
        ranked = sorted(
            available,
            key=lambda item: item["finite_horizon_transfer"]["horizons"][str(horizon)]["G_h_norm"],
        )
        rankings[str(horizon)] = [
            {
                "rank": index + 1,
                "checkpoint": item["checkpoint"],
                "G_h_norm": item["finite_horizon_transfer"]["horizons"][str(horizon)]["G_h_norm"],
                "ranking_scope": "diagnostic transfer-gain proxy; not held-out planning risk",
            }
            for index, item in enumerate(ranked)
        ]
    rejected = sum(item["permission"]["status"] != "permitted" for item in reports)
    counterexample = spectrum_alignment_commutator_counterexample()
    counterexample.update({
        "same_declared_prediction_mse": True,
        "same_declared_action_inverse_risk": True,
        "same_action_norm": True,
        "mse_and_action_risk_status": "declared_matched_nuisance_in_synthetic_counterexample; not learned evidence",
        "horizons": {
            str(h): {
                "left": float(0.9 ** (h - 1)),
                "right": float(0.4 ** (h - 1)),
                "different": bool(not np.isclose(0.9 ** (h - 1), 0.4 ** (h - 1))),
            }
            for h in horizons
        },
    })
    tool_status = "validated_schema_and_fail_closed" if reports and rejected == len(reports) else "partial"
    external = _external_gain_gate(results_root)
    return {
        "artifact": "best_paper_readiness_audit",
        "protocol_version": "best_paper_readiness_audit.v1",
        "source": {
            "record_root": str(record_root),
            "record_count": len(records),
            "load_errors": errors,
            "horizons": list(horizons),
        },
        "transfer_tool": {
            "status": tool_status,
            "record_reports": reports,
            "checkpoint_rankings": rankings,
            "fail_closed_rejection_count": rejected,
            "calibration_interval": {
                "status": "blocked_missing_independent_calibration",
                "reason": "checkpoint records contain no independent calibration residual interval",
            },
        },
        "theory_requirement": {
            "status": "validated_synthetic_counterexample",
            "counterexample": counterexample,
            "tightness_example": transfer_bound_tightness_example(),
            "claim_boundary": "equal declared nuisance factors identify neither G_h nor learned planning transfer",
        },
        "independent_real_gain": external,
        "best_paper_gate": {
            "new_reproducible_law": "pass_synthetic_boundary_not_learned_generalization",
            "reusable_tool": tool_status,
            "independent_real_gain": external["status"],
            "overall_status": "not_ready_without_independent_real_gain_and_cross_environment_calibration",
        },
        "claim_boundary": (
            "This artifact establishes a reusable G_h/risk-gate implementation and a finite-dimensional counterexample. "
            "It does not establish that G_h predicts learned planning risk until independent calibration and an external "
            "non-privileged held-out gain are supplied."
        ),
    }


def markdown(report: dict[str, Any]) -> str:
    gate = report["best_paper_gate"]
    tool = report["transfer_tool"]
    external = report["independent_real_gain"]
    lines = [
        "# Best-Paper readiness audit",
        "",
        f"**Overall status:** `{gate['overall_status']}`",
        "",
        "| Requirement | Status |",
        "|---|---|",
        f"| reproducible transfer law | {gate['new_reproducible_law']} |",
        f"| reusable G_h/risk tool | {gate['reusable_tool']} |",
        f"| independent real gain | {gate['independent_real_gain']} |",
        "",
        "## Tool audit",
        "",
        f"Records: {report['source']['record_count']}; fail-closed rejections: {tool['fail_closed_rejection_count']}",
        "",
        "The tool ranks checkpoints by `||G_h||` as a diagnostic proxy and rejects learned action release when independent gates are absent.",
        "",
        "## Counterexample",
        "",
        "Same declared spectrum, covariance, commutator, prediction MSE, and action inverse risk still yield different finite-horizon `G_h`.",
        "",
        "## Independent gain gate",
        "",
        f"Status: `{external['status']}`; valid explicit artifacts: {external['valid_artifact_count']}",
        "",
        report["claim_boundary"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.record_root, args.results_root)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "best_paper_readiness_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "best_paper_readiness_audit.md").write_text(markdown(report))
    print(json.dumps(report["best_paper_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
