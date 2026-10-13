"""Produce the reader-facing coverage matrix for the Oral experiment package.

This command audits the *evidence required by the protocol*; it does not
impute missing learned-model fields and it never turns a positive local
quantity into a cross-environment claim.  It is deliberately separate from
the matched selector so that a panel can be complete in schema while the
conditional estimand remains non-identifiable.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .run_matched_learned_transfer_benchmark import _load, audit
    from .run_oral_level_unified_diagnostic_audit import (
        HORIZONS,
        REQUIRED_DIAGNOSTICS,
        _coverage,
        _normalise_panel_row,
    )
except ImportError:  # pragma: no cover
    from run_matched_learned_transfer_benchmark import _load, audit  # type: ignore
    from run_oral_level_unified_diagnostic_audit import (  # type: ignore
        HORIZONS,
        REQUIRED_DIAGNOSTICS,
        _coverage,
        _normalise_panel_row,
    )


REQUIRED_DESIGN = {
    "architectures": 3,
    "seeds": 5,
    "environments": 4,
    "snapshots_per_architecture_seed": 6,
    "horizons": list(HORIZONS),
}


def _read(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _paths(roots: list[Path]) -> list[Path]:
    return sorted({path for root in roots for path in root.rglob("checkpoint_record.json")})


def _load_payloads(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    payloads: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        payload = _read(path)
        if payload is None:
            failures.append({"path": str(path), "reason": "invalid_json"})
            continue
        payload["_audit_record_path"] = str(path)
        payloads.append(payload)
    return payloads, failures


def _design(rows: list[dict[str, Any]]) -> dict[str, Any]:
    architectures = sorted({str(row["architecture"]) for row in rows})
    environments = sorted({str(row["environment"]) for row in rows})
    seeds = sorted({str(row["seed"]) for row in rows})
    key_counts = Counter((str(row["architecture"]), str(row["seed"])) for row in rows)
    horizon_support = sorted({int(h) for row in rows for h in row.get("horizons", [])})
    by_environment = {
        environment: {
            "records": sum(row["environment"] == environment for row in rows),
            "architectures": sorted({row["architecture"] for row in rows if row["environment"] == environment}),
            "seeds": sorted({row["seed"] for row in rows if row["environment"] == environment}),
        }
        for environment in environments
    }
    checks = {
        "architectures": len(architectures) >= REQUIRED_DESIGN["architectures"],
        "seeds": len(seeds) >= REQUIRED_DESIGN["seeds"],
        "environments": len(environments) >= REQUIRED_DESIGN["environments"],
        "snapshots_per_architecture_seed": bool(key_counts)
        and min(key_counts.values()) >= REQUIRED_DESIGN["snapshots_per_architecture_seed"],
        "horizons": all(h in horizon_support for h in HORIZONS),
    }
    return {
        "required": REQUIRED_DESIGN,
        "observed": {
            "records": len(rows),
            "architectures": architectures,
            "environments": environments,
            "seeds": seeds,
            "horizon_support": horizon_support,
            "snapshot_counts": {f"{a}/{s}": n for (a, s), n in sorted(key_counts.items())},
            "by_environment": by_environment,
        },
        "checks": checks,
        "status": "pass" if all(checks.values()) else "blocked",
    }


def _metric_matrix(payloads: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Reuse the single source of truth for metric extraction, then add the
    # strata needed to audit the requested architecture/environment design.
    coverage = _coverage(payloads)
    by_env: dict[str, dict[str, Any]] = {}
    for environment in sorted({row["environment"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["environment"] == environment]
        env_payloads = [payloads[i] for i in indices]
        env_cov = _coverage(env_payloads)["required_diagnostics"]
        by_env[environment] = {
            name: {
                "present": item["present"],
                "total": len(env_payloads),
                "coverage": item["coverage"],
                "sources": item["sources"],
                "status": item["status"],
            }
            for name, item in env_cov.items()
        }
    return {"all_records": coverage, "by_environment": by_env}


def _strict_match(paths: list[Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            records.append(_load(path, require_external=True))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({"path": str(path), "reason": str(exc)})
    if len(records) < 2:
        return {"status": "not_estimable", "records": len(records), "load_failures": failures}
    report = audit(records, require_external=True)
    summary = report.get("summary", {})
    return {
        "status": summary.get("status", "unknown"),
        "records": len(records),
        "load_failures": failures,
        "identifiable_strata": summary.get("identifiable_strata", 0),
        "total_strata": summary.get("total_strata", 0),
        "identifiable_environments": summary.get("identifiable_environments", []),
        "identifiable_environment_count": summary.get("identifiable_environment_count", 0),
        "primary_delta": summary.get("primary_delta_risk_mse_minus_full"),
        "claim_ready_for_positive_delta": summary.get("claim_ready_for_positive_delta", False),
        "claim_boundary": report.get("claim_boundary"),
    }


def _artifact(path: Path, *, expected_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    payload = _read(path)
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if payload is None:
        result["status"] = "missing_or_unreadable"
        return result
    result["status"] = "available"
    result["top_level_keys"] = sorted(payload.keys())
    result["expected_fields"] = {
        key: (key in payload) for key in expected_keys
    }
    for key in ("summary", "protocol", "claim_boundary"):
        if key in payload:
            value = payload[key]
            result[key] = value if key != "claim_boundary" else value
    return result


def _benchmark_summary(path: Path) -> dict[str, Any]:
    payload = _read(path)
    if payload is None:
        return {"path": str(path), "status": "missing_or_unreadable"}
    summary = payload.get("summary", {})
    return {
        "path": str(path),
        "status": summary.get("status", "available"),
        "mse_tolerance": payload.get("protocol", {}).get("mse_tolerance"),
        "spectrum_tolerance": payload.get("protocol", {}).get("spectrum_tolerance"),
        "action_risk_tolerance": payload.get("protocol", {}).get("action_risk_tolerance"),
        "identifiable_strata": summary.get("identifiable_strata", 0),
        "total_strata": summary.get("total_strata", 0),
        "identifiable_environments": summary.get("identifiable_environments", []),
        "claim_ready_for_positive_delta": summary.get("claim_ready_for_positive_delta", False),
        "gating_reasons": [
            {"training_stage": row.get("training_stage"), "reason": row.get("reason_if_gated")}
            for row in payload.get("strata", [])
            if row.get("reason_if_gated")
        ],
    }


def _statistical_gates(panel_summary: Path | None, strict: dict[str, Any]) -> dict[str, Any]:
    result = {
        "learned_strict_match": {
            "status": "available" if strict.get("claim_ready_for_positive_delta") else "blocked",
            "claim_ready": bool(strict.get("claim_ready_for_positive_delta")),
        },
        "paired_bootstrap": "blocked_no_formal_learned_matched_strata",
        "sign_flip_permutation": "blocked_no_formal_learned_matched_strata",
        "holm_correction": "blocked_no_formal_learned_matched_strata",
    }
    if panel_summary and panel_summary.exists():
        payload = _read(panel_summary)
        if isinstance(payload, dict):
            families = payload.get("family_summary", [])
            result["pointmaze_panel_summary"] = {
                "path": str(panel_summary),
                "family_count": len(families) if isinstance(families, list) else 0,
                "families": families,
                "claim_boundary": payload.get("claim_boundary"),
            }
            result["paired_bootstrap"] = "available_pointmaze_family_summary_only"
            result["sign_flip_permutation"] = "available_pointmaze_family_summary_only"
            result["holm_correction"] = "not_available_in_pointmaze_family_summary"
    return result


def build_report(record_roots: list[Path], results_root: Path, output: Path, panel_summary: Path | None) -> dict[str, Any]:
    paths = _paths(record_roots)
    payloads, failures = _load_payloads(paths)
    rows = [_normalise_panel_row(payload, Path(payload["_audit_record_path"])) for payload in payloads]
    strict = _strict_match(paths)
    design = _design(rows)
    metric_matrix = _metric_matrix(payloads, rows)
    cross_env_path = results_root / "cross_arch_enriched_20261011" / "cross_environment_audit" / "matched_learned_transfer_benchmark.json"
    nonnormal_path = results_root / "unified_research_protocol_20261012_v2" / "geometry" / "gauge_nonnormal_audit.json"
    synthetic_path = results_root / "fixed_quality_synthetic_intervention_20261013" / "intervention.json"
    fit_window_path = results_root / "pointmaze_common_fit_window_search_20261013" / "fit_window_search.json"
    tolerance_paths = {
        "formal_0.05": results_root / "pointmaze_matched_tolerance05_20261013" / "matched_learned_transfer_benchmark.json",
        "exploratory_0.25": results_root / "pointmaze_matched_tolerance25_exploratory_20261013" / "matched_learned_transfer_benchmark.json",
        "exploratory_0.50": results_root / "pointmaze_matched_tolerance50_exploratory_20261013" / "matched_learned_transfer_benchmark.json",
    }
    return {
        "artifact": "oral_experiment_coverage_audit",
        "protocol_version": "oral_experiment_coverage_audit.v1",
        "input": {
            "record_roots": [str(root) for root in record_roots],
            "record_paths": [str(path) for path in paths],
            "valid_payloads": len(payloads),
            "load_failures": failures,
        },
        "A_unified_diagnostic_comparison": {
            "metric_matrix": metric_matrix,
            "estimand": "Delta R = R(MSE-only) - R(MSE+diagnostic)",
            "matched_learned_estimand": strict,
            "status": "schema_complete_but_estimand_blocked" if any(
                item["status"] == "available" for item in metric_matrix["all_records"]["required_diagnostics"].values()
            ) and not strict.get("claim_ready_for_positive_delta") else "available",
        },
        "B_real_learned_matched_intervention": {
            "design": design,
            "strict_match": strict,
            "training_only_common_window_search": _artifact(
                fit_window_path,
                expected_keys=("protocol", "selected", "candidate_windows"),
            ),
            "tolerance_sensitivity": {
                name: _benchmark_summary(path) for name, path in tolerance_paths.items()
            },
            "diagnosis": (
                "The formal panel has no cross-architecture MSE+spectrum matched stratum. "
                "The training-only common-window search and exploratory tolerance runs are retained "
                "as diagnostics; they do not redefine the preregistered formal gate."
            ),
            "status": "validated_only_if_strict_gate_passes" if strict.get("claim_ready_for_positive_delta") else "non_identifiable_under_current_panel",
        },
        "C_ATM_external_metric": {
            "field": "action_inverse_risk / atm_action_inverse_risk",
            "coverage": metric_matrix["all_records"]["required_diagnostics"]["atm_action_inverse_risk"],
            "status": "available_as_external_field" if metric_matrix["all_records"]["required_diagnostics"]["atm_action_inverse_risk"]["present"] else "missing",
            "claim_boundary": "availability does not establish incremental predictive validity",
        },
        "D_fixed_spectrum_nonnormal_intervention": {
            "nonnormal_audit": _artifact(nonnormal_path, expected_keys=("spectrum_fixed_nonnormal_family", "protocol")),
            "fixed_quality_intervention": _artifact(synthetic_path, expected_keys=("support", "selector_regret", "adversarial_selector_regret")),
            "status": "synthetic_or_controlled_only",
        },
        "E_cross_environment_LOO": {
            "current_panel_design": design,
            "existing_cross_environment_artifact": _artifact(cross_env_path, expected_keys=("summary", "strata", "protocol")),
            "status": "available_only_as_gated_diagnostic" if len(design["observed"]["environments"]) < 4 else "requires_formal_run",
            "leave_one_environment_out": "blocked_until_at_least_two_environments_have_identifiable_matched_strata",
        },
        "statistical_gates": _statistical_gates(panel_summary, strict),
        "overall_status": "blocked_oral_gate_with_partial_validated_components",
        "claim_boundary": (
            "The audit separates field availability, matched-identifiability, and independent planning consequence. "
            "It does not promote a PointMaze-only or representation-conditioned positive result to a universal claim."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    design = report["B_real_learned_matched_intervention"]["design"]
    matrix = report["A_unified_diagnostic_comparison"]["metric_matrix"]["all_records"]["required_diagnostics"]
    lines = [
        "# Oral experiment coverage audit",
        "",
        f"**Overall status:** `{report['overall_status']}`",
        f"**Strict learned matched status:** `{report['B_real_learned_matched_intervention']['status']}`",
        "",
        "## Design coverage",
        "",
        f"Observed architectures={len(design['observed']['architectures'])}, environments={len(design['observed']['environments'])}, seeds={len(design['observed']['seeds'])}, horizon support={design['observed']['horizon_support']}.",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for name, passed in design["checks"].items():
        lines.append(f"| {name} | {'PASS' if passed else 'BLOCKED'} |")
    lines.extend(["", "## Unified diagnostic field coverage", "", "| Field | Present | Total | Source | Status |", "|---|---:|---:|---|---|"])
    for name, item in matrix.items():
        lines.append(f"| {name} | {item['present']} | {item['present'] + item['missing']} | {item['sources']} | {item['status']} |")
    lines.extend([
        "", "## Scientific boundary", "", report["claim_boundary"], "",
        "A field marked available is a schema/provenance result, not evidence that the metric predicts held-out planning risk.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", action="append", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--panel-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.record_root, args.results_root, args.output, args.panel_summary)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "oral_experiment_coverage_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "oral_experiment_coverage_audit.md").write_text(_markdown(report))
    print(json.dumps({
        "output": str(args.output),
        "overall_status": report["overall_status"],
        "strict_status": report["B_real_learned_matched_intervention"]["status"],
        "design_status": report["B_real_learned_matched_intervention"]["design"]["status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
