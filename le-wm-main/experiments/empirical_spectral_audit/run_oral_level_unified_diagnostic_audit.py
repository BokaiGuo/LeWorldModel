"""Run the frozen Oral-level diagnostic audit over the learned-model panel.

This is an evidence aggregator, not a positive-result generator.  It combines
the immutable checkpoint records with the existing strict matched audit and
the already-computed controlled/synthetic artifacts.  Missing fields and
unidentifiable strata remain explicit.  No held-out planning outcome is read
when constructing a selector.

The output is deliberately useful in two states:

* with a future panel that contains identifiable matched strata, it reports
  paired bootstrap, exact sign-flip permutation tests, and Holm correction;
* with the current panel, it reports the protocol coverage and the exact gate
  that prevents those tests from being interpreted as a learned-model result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:  # package execution
    from .oral_evidence_protocol import validate_learned_checkpoint_panel
    from .run_matched_learned_transfer_benchmark import _load, audit
except ImportError:  # direct script execution from experiments directory
    from oral_evidence_protocol import validate_learned_checkpoint_panel  # type: ignore[no-redef]
    from run_matched_learned_transfer_benchmark import _load, audit  # type: ignore[no-redef]


HORIZONS = (1, 2, 4, 8, 16)
REQUIRED_DIAGNOSTICS = (
    "prediction_mse",
    "latent_variance",
    "covariance_alignment",
    "principal_angle",
    "projector_overlap",
    "commutator",
    "schur_offdiagonal_energy",
    "resolvent_norm",
    "transient_amplification",
    "atm_action_inverse_risk",
    "real_predicted_transition_transfer",
    "reachability_score",
    "candidate_elite_regret",
    "planning_risk",
)

ALIASES = {
    "covariance_alignment": ("covariance_alignment", "alignment", "task_metric_alignment"),
    "principal_angle": ("principal_angle", "principal_angles"),
    "projector_overlap": ("projector_overlap", "spectral_projector_overlap"),
    "commutator": ("commutator", "normalized_commutator", "task_metric_commutator"),
    "schur_offdiagonal_energy": (
        "schur_offdiagonal_energy", "task_metric_schur_offdiagonal_energy",
    ),
    "resolvent_norm": ("resolvent_norm", "pseudospectral_sensitivity"),
    "transient_amplification": (
        "transient_amplification", "task_metric_transient_amplification",
    ),
    # Learned adapter records historically used ``action_inverse_risk`` while
    # the matched-transfer records used the longer ATM name.  These are the
    # same predeclared external action-transfer quantity; omitting the short
    # spelling silently turned an available field into a false coverage gap.
    "atm_action_inverse_risk": (
        "atm_action_inverse_risk", "action_inverse_risk", "inverse_action_risk", "atm",
    ),
    "real_predicted_transition_transfer": (
        "real_predicted_transition_transfer", "transition_transfer", "counterfactual_transfer_error",
    ),
    "reachability_score": ("reachability_score", "reachable_score"),
    "candidate_elite_regret": ("candidate_elite_regret", "elite_regret"),
    "planning_risk": ("planning_risk", "heldout_planning_loss"),
}


_FIT_VARIANCE_CACHE: dict[str, float | None] = {}
_FIT_HORIZON_CACHE: dict[str, list[int]] = {}


def _resolve_provenance_path(value: Any, record_path: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    # Relative provenance paths in the panel are rooted at the experiment
    # checkout, not at the record directory.
    candidates = [Path.cwd() / path, record_path.parent / path]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _fit_features(payload: dict[str, Any], record_path: Path) -> tuple[float | None, list[int]]:
    record = payload.get("record", {})
    provenance = record.get("provenance", {}) if isinstance(record, dict) else {}
    fit_path = _resolve_provenance_path(provenance.get("fit_export"), record_path)
    if fit_path is None:
        return None, []
    cache_key = str(fit_path)
    if cache_key in _FIT_VARIANCE_CACHE:
        return _FIT_VARIANCE_CACHE[cache_key], _FIT_HORIZON_CACHE.get(cache_key, [])
    variance: float | None = None
    horizons: list[int] = []
    fit = _read_json(fit_path)
    if isinstance(fit, dict):
        raw_horizons = fit.get("horizons")
        if isinstance(raw_horizons, list):
            horizons = sorted({int(value) for value in raw_horizons if isinstance(value, (int, float))})
        rows = fit.get("enriched_records")
        if isinstance(rows, list):
            latents = [item.get("source_latent") for item in rows if isinstance(item, dict)]
            if latents:
                try:
                    matrix = np.asarray(latents, dtype=np.float64)
                    if matrix.ndim == 2 and matrix.shape[0] >= 2 and np.isfinite(matrix).all():
                        # Declared definition: mean coordinate variance of the
                        # train-only source latent, never the held-out outcome.
                        variance = float(np.var(matrix, axis=0, ddof=1).mean())
                except (TypeError, ValueError, FloatingPointError):
                    variance = None
    _FIT_VARIANCE_CACHE[cache_key] = variance
    _FIT_HORIZON_CACHE[cache_key] = horizons
    return variance, horizons


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value(row: dict[str, Any], name: str) -> Any:
    diagnostics = row.get("diagnostics", {})
    heldout = row.get("heldout", {})
    record = row.get("record", {}) if isinstance(row.get("record"), dict) else {}
    nested = record.get("diagnostics", {}) if isinstance(record.get("diagnostics"), dict) else {}
    candidates = list(ALIASES.get(name, (name,)))
    if name == "planning_risk":
        candidates = list(candidates) + ["planning_loss"]
    for key in candidates:
        for source in (row, diagnostics, heldout, record, nested):
            if isinstance(source, dict) and key in source:
                value = source[key]
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    return value
                if isinstance(value, list) and value:
                    return value
    derived = _derived_value(row, name)
    if derived is not None:
        return derived
    return None


def _transfer_operator(row: dict[str, Any]) -> dict[str, Any] | None:
    record = row.get("record", {}) if isinstance(row.get("record"), dict) else {}
    diagnostics = record.get("diagnostics", {}) if isinstance(record.get("diagnostics"), dict) else {}
    operator = diagnostics.get("transfer_operator")
    return operator if isinstance(operator, dict) else None


def _derived_value(row: dict[str, Any], name: str) -> float | None:
    """Compute only operator-level quantities that are fully determined by A/B.

    These are explicitly marked as derived proxies in the coverage report.  In
    particular, they do not replace an external ATM intervention or a held-out
    real/predicted successor comparison.
    """
    if name == "latent_variance":
        value = row.get("_latent_variance")
        return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None
    if name == "candidate_elite_regret":
        details = row.get("heldout", {}).get("planning_details", {})
        if isinstance(details, dict):
            oracle = details.get("mean_oracle_progress")
            chosen = details.get("mean_chosen_progress")
            if all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in (oracle, chosen)):
                # Positive means the selected candidate trails the oracle.
                return float(oracle) - float(chosen)
        return None
    operator = _transfer_operator(row)
    if not operator:
        return None
    try:
        A = np.asarray(operator["A"], dtype=np.float64)
        B = np.asarray(operator["B"], dtype=np.float64)
        if A.ndim != 2 or B.ndim != 2 or A.shape[0] != A.shape[1] or B.shape[0] != A.shape[0]:
            return None
        if name == "resolvent_norm":
            # Fixed unit-circle grid; the grid is a declared diagnostic, not a
            # data-dependent maximization over the held-out outcome.
            values = []
            identity = np.eye(A.shape[0], dtype=np.complex128)
            for theta in np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False):
                values.append(float(np.linalg.norm(np.linalg.pinv(np.exp(1j * theta) * identity - A), 2)))
            return max(values)
        if name == "reachability_score":
            gramian = np.zeros((A.shape[0], A.shape[0]), dtype=np.float64)
            power = np.eye(A.shape[0], dtype=np.float64)
            for _ in range(4):
                response = power @ B
                gramian += response @ response.T
                power = A @ power
            return float(np.trace(gramian))
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None
    return None


def _value_source(row: dict[str, Any], name: str) -> str:
    diagnostics = row.get("diagnostics", {})
    heldout = row.get("heldout", {})
    record = row.get("record", {}) if isinstance(row.get("record"), dict) else {}
    nested = record.get("diagnostics", {}) if isinstance(record.get("diagnostics"), dict) else {}
    keys = list(ALIASES.get(name, (name,)))
    if name == "planning_risk":
        keys.append("planning_loss")
    for key in keys:
        for source in (row, diagnostics, heldout, record, nested):
            if isinstance(source, dict) and key in source and source[key] is not None:
                return "explicit"
    derived = _derived_value(row, name)
    if derived is not None:
        if name == "latent_variance":
            return "derived_fit_train"
        if name == "candidate_elite_regret":
            return "derived_heldout_outcome"
        return "derived_operator_proxy"
    return "missing"


def _normalise_panel_row(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    record = payload.get("record", {})
    protocol = payload.get("protocol", {})
    factors = record.get("model_factors", {}) if isinstance(record, dict) else {}
    diagnostics = record.get("diagnostics", {}) if isinstance(record, dict) else {}
    heldout = record.get("heldout", {}) if isinstance(record, dict) else {}
    architecture_value = record.get("architecture")
    if not architecture_value and isinstance(factors, dict):
        architecture_value = factors.get("adapter_architecture")
    if not architecture_value:
        group = str(record.get("group", ""))
        group_parts = [part for part in group.split("/") if part]
        architecture_value = next(
            (part for part in reversed(group_parts) if not part.startswith("seed_")),
            "unknown",
        )
    architecture = str(architecture_value)
    environment = str(protocol.get("environment", "unknown"))
    checkpoint = str(record.get("checkpoint", path))
    seed_match = re.search(r"seed[_-]([^/]+)", checkpoint)
    seed = str(factors.get("seed", record.get("seed", seed_match.group(1) if seed_match else "unknown")))
    action_distribution = str(
        factors.get("action_distribution")
        or protocol.get("prediction_quality", {}).get("action_distribution", "missing")
    )
    latent_variance, fit_horizons = _fit_features(payload, path)
    protocol_horizons = protocol.get("horizons", []) if isinstance(protocol, dict) else []
    horizons = fit_horizons or [int(value) for value in protocol_horizons if isinstance(value, (int, float))]
    if not horizons:
        prediction_quality = protocol.get("prediction_quality", {}) if isinstance(protocol, dict) else {}
        if isinstance(prediction_quality, dict) and isinstance(prediction_quality.get("fit_horizon"), (int, float)):
            horizons = [int(prediction_quality["fit_horizon"])]
    row = {
        "architecture": architecture,
        "environment": environment,
        "seed": seed,
        "training_stage": factors.get("training_stage"),
        "latent_dimension": factors.get("latent_dimension"),
        "horizons": sorted(set(horizons)),
        "prediction_mse": diagnostics.get("prediction_mse"),
        "atm_action_inverse_risk": diagnostics.get("atm_action_inverse_risk"),
        "predictor_spectrum": diagnostics.get("predictor_spectrum"),
        "task_metric": {
            key: diagnostics.get(key)
            for key in diagnostics
            if key.startswith("task_metric_")
        },
        "planning_risk": heldout.get("planning_loss"),
        "atm_action_inverse_risk": diagnostics.get("atm_action_inverse_risk", diagnostics.get("action_inverse_risk")),
        "_latent_variance": diagnostics.get("latent_variance", latent_variance),
        "heldout": heldout,
        "checkpoint": checkpoint,
        "_path": str(path),
    }
    return row


def _coverage(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    source_counts: dict[str, dict[str, int]] = {}
    total = len(payloads)
    for name in REQUIRED_DIAGNOSTICS:
        present = 0
        sources = Counter()
        for payload in payloads:
            row = payload.get("record", {})
            if not isinstance(row, dict):
                continue
            record_path = Path(str(payload.get("_audit_record_path", "checkpoint_record.json")))
            candidate = _normalise_panel_row(payload, record_path)
            candidate.update({
                "record": row,
                "diagnostics": row.get("diagnostics", {}),
                "heldout": row.get("heldout", {}),
            })
            if _value(candidate, name) is not None:
                present += 1
                sources[_value_source(candidate, name)] += 1
        counts[name] = present
        source_counts[name] = dict(sorted(sources.items()))
    return {
        "records": total,
        "required_diagnostics": {
            name: {
                "present": counts[name],
                "missing": total - counts[name],
                "coverage": None if total == 0 else counts[name] / total,
                "status": "available" if counts[name] == total else "partial_or_missing",
                "sources": source_counts[name],
            }
            for name in REQUIRED_DIAGNOSTICS
        },
        "interpretation": "coverage is a protocol check; it is not evidence that a field is valid or predictive",
    }


def _panel_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return validate_learned_checkpoint_panel(
        rows,
        minimum_architectures=3,
        minimum_seeds=5,
        minimum_environments=4,
        minimum_snapshots=6,
        horizons=HORIZONS,
    )


def _summarise_artifact(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        result["status"] = "not_available"
        return result
    result["sha256"] = _sha256(path)
    if payload is None:
        result["status"] = "unreadable_json"
        return result
    result["status"] = "available"
    result["top_level_keys"] = sorted(payload)[:40]
    if isinstance(payload.get("summary"), dict):
        summary = payload["summary"]
        result["summary_status"] = summary.get("status")
        for key in (
            "identifiable_strata", "total_strata", "environment_count", "architecture_count",
            "seed_count", "identifiable_environment_count",
        ):
            if key in summary:
                result[key] = summary[key]
    if "protocol" in payload and isinstance(payload["protocol"], dict):
        result["protocol_claim_boundary"] = payload["protocol"].get("claim_boundary")
    if "claim_boundary" in payload:
        result["claim_boundary"] = payload["claim_boundary"]
    for key in ("support", "selector_regret", "adversarial_selector_regret", "aggregate_association"):
        if key in payload:
            result[key] = payload[key]
    protocol = payload.get("protocol")
    if isinstance(protocol, dict):
        for key in ("horizons", "synthetic_vector_world_only", "rollout_and_planning_available"):
            if key in protocol:
                result[key] = protocol[key]
    return result


def _discover_artifacts(results_root: Path) -> dict[str, Path]:
    return {
        "strict_matched_current": results_root / "learned_transfer_operator_panel_20261013" / "matched_audit" / "matched_learned_transfer_benchmark.json",
        "strict_matched_previous": results_root / "cross_arch_enriched_snapshots_20261011" / "matched_audit" / "matched_learned_transfer_benchmark.json",
        "strict_matched_cross_environment": results_root / "cross_arch_enriched_snapshots_20261011" / "cross_environment_audit" / "matched_learned_transfer_benchmark.json",
        "controlled_alignment": results_root / "unified_research_protocol_20261012_v2" / "controlled_alignment" / "diagnostic.json",
        "gauge_nonnormal": results_root / "unified_research_protocol_20261012_v2" / "geometry" / "gauge_nonnormal_audit.json",
        "six_goal_matrix": results_root / "unified_research_protocol_20261012_v2" / "claim_matrix" / "six_goal_evidence_matrix.json",
        "fixed_quality_synthetic_intervention": results_root / "fixed_quality_synthetic_intervention_20261013" / "intervention.json",
        "horizon_operator_geometry": results_root / "horizon_operator_geometry_synthetic_20261003" / "horizon_operator_geometry.json",
        "frozen_nonnormal_control": results_root / "nonnormality_control_frozen_20261003" / "nonnormality_control_frozen.json",
        "frozen_geometry_error_association": results_root / "geometry_error_association_frozen_20261003" / "geometry_error_association_frozen.json",
    }


def _nonnormal_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not payload:
        return {"status": "not_available", "path": str(path)}
    family = payload.get("spectrum_fixed_nonnormal_family", [])
    if not isinstance(family, list):
        return {"status": "available_but_unparsed", "path": str(path)}
    return {
        "status": "synthetic_only",
        "path": str(path),
        "fixed_spectrum": bool(payload.get("protocol", {}).get("spectrum_fixed_nonnormal_family")),
        "family_size": len(family),
        "alpha_values": [item.get("alpha") for item in family if isinstance(item, dict)],
        "nonnormality_values": [item.get("nonnormality") for item in family if isinstance(item, dict)],
        "metric": "nonnormality; finite-horizon transient gain is not inferred when absent",
        "claim_boundary": "controlled/synthetic evidence only; not learned-model planning evidence",
    }


def _permutation_and_bootstrap(values: Iterable[float], seed: int) -> dict[str, Any]:
    """Exact sign-flip audit for paired improvements, when matched rows exist."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"status": "not_identifiable", "n": 0}
    if array.size > 20:
        # Exact enumeration would be unnecessarily expensive; deterministic
        # Monte Carlo is retained as an explicit approximation.
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(20000, array.size))
        null = (signs * array[None, :]).mean(axis=1)
        method = "monte_carlo_sign_flip"
    else:
        signs = np.asarray(
            [[1.0 if mask & (1 << index) else -1.0 for index in range(array.size)]
             for mask in range(1 << array.size)], dtype=np.float64,
        )
        null = (signs * array[None, :]).mean(axis=1)
        method = "exact_sign_flip"
    observed = float(array.mean())
    p = float(np.mean(np.abs(null) >= abs(observed)))
    rng = np.random.default_rng(seed + 1)
    means = rng.choice(array, size=(4000, array.size), replace=True).mean(axis=1)
    return {
        "status": "available",
        "n": int(array.size),
        "mean": observed,
        "bootstrap_ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "sign_flip_p": p,
        "method": method,
    }


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        corrected = min(1.0, (len(ordered) - index) * value)
        running = max(running, corrected)
        adjusted[name] = running
    return adjusted


def run_audit(record_root: Path, results_root: Path, output: Path) -> dict[str, Any]:
    record_paths = sorted(record_root.rglob("checkpoint_record.json"))
    if not record_paths:
        raise FileNotFoundError(f"no checkpoint_record.json found under {record_root}")
    payloads = []
    valid_pairs: list[tuple[dict[str, Any], Path]] = []
    loaded = []
    load_errors = []
    for path in record_paths:
        payload = _read_json(path)
        if payload is None:
            load_errors.append({"path": str(path), "reason": "invalid_json"})
            continue
        payloads.append(payload)
        payload["_audit_record_path"] = str(path)
        valid_pairs.append((payload, path))
        try:
            loaded.append(_load(path))
        except (ValueError, KeyError, TypeError) as exc:
            load_errors.append({"path": str(path), "reason": str(exc)})
    rows = [_normalise_panel_row(payload, path) for payload, path in valid_pairs]
    artifacts = _discover_artifacts(results_root)
    matched = audit(loaded) if loaded else {"summary": {"status": "not_available"}, "strata": []}
    matched_summary = matched.get("summary", {})
    # The strict runner reports local selector quantities for any identifiable
    # stratum.  Formal cross-environment inference requires replication in at
    # least two independent strata and two environments; otherwise all
    # resampling statistics are intentionally gated, even when n=1 quantities
    # happen to be numerically computable.
    formal_matched = bool(
        matched_summary.get("identifiable_strata", 0) >= 2
        and matched_summary.get("identifiable_environment_count", 0) >= 2
    )
    paired = {}
    p_values: dict[str, float] = {}
    for metric, item in matched_summary.get("selection_regret", {}).items():
        if not formal_matched:
            continue
        values = []
        for stratum in matched.get("strata", []):
            selector = stratum.get("selectors", {}).get(metric)
            if isinstance(selector, dict) and isinstance(selector.get("regret"), (int, float)):
                values.append(float(selector["regret"]))
        paired[metric] = _permutation_and_bootstrap(values, 20261013)
        if paired[metric].get("status") == "available":
            p_values[metric] = float(paired[metric]["sign_flip_p"])
    for metric, item in paired.items():
        if p_values:
            item["holm_p"] = _holm(p_values).get(metric)
    output.mkdir(parents=True, exist_ok=True)
    matched_path = output / "matched_learned_transfer_benchmark.json"
    matched_path.write_text(json.dumps(matched, indent=2, sort_keys=True) + "\n")
    artifacts = dict(artifacts)
    artifacts["strict_matched_current"] = matched_path
    report = {
        "artifact": "oral_level_unified_diagnostic_audit",
        "protocol_version": "oral_level_unified_diagnostic_audit.v1",
        "source": {
            "record_root": str(record_root),
            "record_count": len(record_paths),
            "loaded_count": len(loaded),
            "load_errors": load_errors,
        },
        "panel": {
            "coverage": _coverage(payloads),
            "design_gate": _panel_gate(rows),
            "inventory": {
                "architectures": sorted({row["architecture"] for row in rows}),
                "environments": sorted({row["environment"] for row in rows}),
                "seeds": sorted({row["seed"] for row in rows}),
                "snapshots_per_architecture": dict(Counter(row["architecture"] for row in rows)),
            },
        },
        "strict_matched_audit": {
            "summary": matched_summary,
            "paired_statistics": paired if paired else {"status": "blocked_no_identifiable_matched_strata"},
            "claim_status": (
                "validated_conditional_only" if formal_matched else "non_identifiable"
            ),
        },
        "diagnostic_artifacts": {
            name: _summarise_artifact(path) for name, path in artifacts.items()
        },
        "non_normality": _nonnormal_summary(artifacts["gauge_nonnormal"]),
        "required_oral_matrix": {
            "unified_diagnostic_comparison": "partial_if_fields_missing; matched outcome required for predictive claims",
            "real_learned_matched_intervention": (
                "validated_conditional_only" if formal_matched else "non_identifiable"
            ),
            "atm_external_metric": "not_available" if _coverage(payloads)["required_diagnostics"]["atm_action_inverse_risk"]["missing"] else "available",
            "fixed_spectrum_nonnormal_intervention": _nonnormal_summary(artifacts["gauge_nonnormal"])["status"],
            "leave_one_environment_out": (
                "available" if formal_matched else "blocked_single_environment_or_no_matched_strata"
            ),
            "paired_bootstrap_permutation_holm": "available" if formal_matched and paired and any(item.get("status") == "available" for item in paired.values()) else "blocked_insufficient_independent_matched_strata",
        },
        "claim_boundary": (
            f"The current learned panel contains {len(set(row['environment'] for row in rows))} environment(s) and "
            "has a substantial checkpoint inventory, but the strict matched estimand is non-identifiable because "
            "no replicated cross-architecture, cross-seed MSE+spectrum matched strata are available across the "
            "required independent environments. Controlled geometry and non-normality artifacts remain "
            "synthetic/diagnostic evidence only."
        ),
    }
    json_path = output / "oral_level_unified_diagnostic_audit.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _markdown(report: dict[str, Any]) -> str:
    panel = report["panel"]
    summary = report["strict_matched_audit"]["summary"]
    lines = [
        "# Oral-level unified diagnostic audit",
        "",
        f"**Strict matched status:** `{summary.get('status', 'not_available')}`  ",
        f"**Matched strata:** {summary.get('identifiable_strata', 0)} / {summary.get('total_strata', 0)}  ",
        f"**Panel records:** {report['source']['loaded_count']} / {report['source']['record_count']}",
        "",
        "## Panel design gate",
        "",
        f"`{panel['design_gate']['status']}`; required design: 3 architectures, 5 seeds, 4 environments, 6 snapshots, horizons {HORIZONS}.",
        "",
        "| Inventory | Count |",
        "|---|---:|",
        f"| architectures | {len(panel['inventory']['architectures'])} |",
        f"| environments | {len(panel['inventory']['environments'])} |",
        f"| seeds | {len(panel['inventory']['seeds'])} |",
        f"| records | {report['source']['loaded_count']} |",
        "",
        "## Required diagnostic coverage",
        "",
        "| Field | Present | Missing | Status |",
        "|---|---:|---:|---|",
    ]
    for name, item in panel["coverage"]["required_diagnostics"].items():
        lines.append(f"| {name} | {item['present']} | {item['missing']} | {item['status']} |")
    lines.extend(["", "## Existing artifact gates", "", "| Artifact | Status | Summary |", "|---|---|---|"])
    for name, item in report["diagnostic_artifacts"].items():
        summary_text = item.get("summary_status", item.get("status", "unknown"))
        lines.append(f"| {name} | {item.get('status', 'unknown')} | {summary_text} |")
    lines.extend([
        "",
        "## Scientific boundary",
        "",
        report["claim_boundary"],
        "",
        "The audit preserves negative and non-identifiable outcomes; it does not convert protocol coverage into a planning claim.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(args.record_root, args.results_root, args.output)
    (args.output / "oral_level_unified_diagnostic_audit.md").write_text(_markdown(report))
    print(json.dumps({
        "strict_matched_status": report["strict_matched_audit"]["summary"].get("status"),
        "identifiable_strata": report["strict_matched_audit"]["summary"].get("identifiable_strata"),
        "panel_gate": report["panel"]["design_gate"]["status"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
