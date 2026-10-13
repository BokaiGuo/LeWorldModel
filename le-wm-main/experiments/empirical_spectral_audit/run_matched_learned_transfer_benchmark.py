"""Strict matched-quality benchmark for learned covariance--predictor diagnostics.

This runner consumes ``real_checkpoint_selection.v1`` records and evaluates the
pre-declared task/action diagnostics only after fixing the nuisance factors
that otherwise make a model-selection claim uninterpretable:

* pairwise prediction MSE (within a declared tolerance);
* latent dimension;
* the complete predictor eigenvalue fingerprint;
* training stage; and
* the exact action-distribution declaration.

The estimand is conditional transfer-risk association within an environment
stratum.  Metrics are read from the diagnostic split and held-out planning
loss is used only as an outcome.  A record pool with no matched pair is not
silently pooled with unmatched records.  In particular, this script cannot
turn a single architecture, a single environment, or a spectrum-mismatched
collection into a universal model-selection result.

The current checkpoint schema contains task-metric alignment, commutator, and
transient amplification, but older exports may not contain the task-metric
Schur term.  The primary four-metric audit is therefore fail-closed when that
field is absent; a separate three-metric diagnostic is reported only as
``incomplete_schur`` and is never promoted to the primary claim.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "task_metric_alignment",
    "task_metric_commutator",
    "task_metric_schur_offdiagonal_energy",
    "task_metric_transient_amplification",
)
REQUIRED_FACTORS = ("latent_dimension", "training_stage", "action_distribution")


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _stage(record: dict[str, Any]) -> int:
    factors = record.get("model_factors", {})
    if factors.get("training_stage") is not None:
        return int(factors["training_stage"])
    match = re.search(r"epoch[_/](\d+)", str(record.get("checkpoint", "")))
    if match is None:
        raise ValueError(f"cannot infer training stage from {record.get('checkpoint')!r}")
    return int(match.group(1))


def _architecture(record: dict[str, Any]) -> str:
    group = str(record.get("group", ""))
    parts = group.split("/")
    if len(parts) >= 2:
        return parts[-2]
    return str(record.get("architecture", "unknown"))


def _seed(record: dict[str, Any]) -> str:
    match = re.search(r"seed[_-]([^/]+)", str(record.get("group", record.get("checkpoint", ""))))
    return match.group(1) if match else "unknown"


def _environment(record: dict[str, Any]) -> str:
    protocol = record.get("protocol", {})
    return str(protocol.get("environment", str(record.get("group", "unknown")).split("/")[0]))


def _action_distribution(record: dict[str, Any]) -> str:
    factors = record.get("model_factors", {})
    if factors.get("action_distribution") is not None:
        return str(factors["action_distribution"])
    protocol = record.get("protocol", {})
    prediction_quality = protocol.get("prediction_quality", {})
    return str(prediction_quality.get("action_distribution", "missing"))


def _spectrum(record: dict[str, Any]) -> np.ndarray:
    values = record.get("diagnostics", {}).get("predictor_spectrum")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{record.get('checkpoint')!r} lacks predictor_spectrum")
    try:
        array = np.asarray([complex(float(item["real"]), float(item["imag"])) for item in values])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid predictor_spectrum in {record.get('checkpoint')!r}") from exc
    return array[np.lexsort((array.imag, array.real))]


def _spectrum_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-12)
    return float(np.max(np.abs(left - right)) / scale)


def _symmetric_relative_distance(left: float, right: float) -> float:
    """Return the bounded, pairwise relative distance used for MSE matching."""
    denominator = max((abs(left) + abs(right)) / 2.0, 1e-12)
    return float(abs(left - right) / denominator)


def _maximum_compatible_clique(
    candidates: list[dict[str, Any]],
    *,
    mse_tolerance: float,
    spectrum_tolerance: float,
    action_risk_tolerance: float | None = None,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Find a deterministic best matched clique under pairwise rules.

    Matching is intentionally pairwise.  A global best-MSE anchor is not a
    valid reference because it can be incompatible with a pair of candidates
    that are mutually matched.  The candidate pools in this audit are small,
    so a Bron--Kerbosch search is both exact and cheap.  A clique satisfying
    the declared cross-architecture and cross-seed stratum requirements is
    preferred to a larger single-architecture clique; among eligible cliques,
    the largest one is selected.
    """
    count = len(candidates)
    adjacency = [set() for _ in range(count)]
    edges: list[tuple[int, int]] = []
    mse = [float(item["diagnostics"]["prediction_mse"]) for item in candidates]
    action_risk = [float(item["model_factors"].get("action_inverse_risk", np.nan)) for item in candidates]
    for left in range(count):
        for right in range(left + 1, count):
            compatible = (
                _symmetric_relative_distance(mse[left], mse[right]) <= mse_tolerance
                and _spectrum_distance(candidates[left]["_spectrum"], candidates[right]["_spectrum"])
                <= spectrum_tolerance
            )
            if action_risk_tolerance is not None:
                compatible = compatible and _finite(action_risk[left]) and _finite(action_risk[right]) and (
                    _symmetric_relative_distance(action_risk[left], action_risk[right]) <= action_risk_tolerance
                )
            if compatible:
                adjacency[left].add(right)
                adjacency[right].add(left)
                edges.append((left, right))

    best: tuple[int, ...] = ()

    def better(candidate: tuple[int, ...], incumbent: tuple[int, ...]) -> bool:
        candidate_architectures = len({candidates[index]["_architecture"] for index in candidate})
        incumbent_architectures = len({candidates[index]["_architecture"] for index in incumbent})
        candidate_seeds = len({candidates[index]["_seed"] for index in candidate})
        incumbent_seeds = len({candidates[index]["_seed"] for index in incumbent})
        candidate_is_diverse = len(candidate) >= 2 and candidate_architectures >= 2 and candidate_seeds >= 2
        incumbent_is_diverse = len(incumbent) >= 2 and incumbent_architectures >= 2 and incumbent_seeds >= 2
        if candidate_is_diverse != incumbent_is_diverse:
            return candidate_is_diverse
        if len(candidate) != len(incumbent):
            return len(candidate) > len(incumbent)
        if candidate_architectures != incumbent_architectures:
            return candidate_architectures > incumbent_architectures
        if candidate_seeds != incumbent_seeds:
            return candidate_seeds > incumbent_seeds
        # Stable final tie-break: prefer lower average prediction error and
        # then lexicographically earlier source checkpoints.
        candidate_mse = np.mean([mse[index] for index in candidate]) if candidate else float("inf")
        incumbent_mse = np.mean([mse[index] for index in incumbent]) if incumbent else float("inf")
        if not np.isclose(candidate_mse, incumbent_mse):
            return candidate_mse < incumbent_mse
        return tuple(str(candidates[index]["checkpoint"]) for index in candidate) < tuple(
            str(candidates[index]["checkpoint"]) for index in incumbent
        )

    def bron_kerbosch(current: tuple[int, ...], possible: set[int]) -> None:
        nonlocal best
        if not possible:
            ordered = tuple(sorted(current))
            if better(ordered, best):
                best = ordered
            return
        # A non-diverse incumbent cannot prune by clique size: a smaller
        # diverse clique may still be the valid estimand.  Once a diverse
        # incumbent exists, the usual cardinality bound is safe.
        best_architectures = len({candidates[index]["_architecture"] for index in best})
        best_seeds = len({candidates[index]["_seed"] for index in best})
        best_is_diverse = len(best) >= 2 and best_architectures >= 2 and best_seeds >= 2
        if best_is_diverse and len(current) + len(possible) < len(best):
            return
        # Deterministic branch order improves reproducibility of the audit.
        for vertex in sorted(tuple(possible)):
            bron_kerbosch(current + (vertex,), possible & adjacency[vertex])
            possible.remove(vertex)

    bron_kerbosch((), set(range(count)))
    return list(best), edges


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(metric: np.ndarray, outcome: np.ndarray) -> float | None:
    if len(metric) < 3 or np.ptp(metric) <= 1e-12 or np.ptp(outcome) <= 1e-12:
        return None
    return float(np.corrcoef(_rank(metric), _rank(outcome))[0, 1])


def _bootstrap(values: list[float], seed: int, draws: int = 4000) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n_strata": 0}
    rng = np.random.default_rng(seed)
    sample = array[rng.integers(0, array.size, size=(draws, array.size))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(sample, 0.025)),
        "ci95_high": float(np.quantile(sample, 0.975)),
        "n_strata": int(array.size),
    }


def _positive_delta_test(values: list[float], seed: int, draws: int = 20000) -> dict[str, Any]:
    """One-sided paired sign-flip and paired bootstrap for Delta R > 0.

    The replication unit is an identifiable matched environment/stage stratum,
    never an episode.  Under the sharp symmetric null, randomly flipping the
    sign of each stratum-level difference gives a conservative small-sample
    test of whether the mean improvement is positive.  The bootstrap interval
    is reported alongside the permutation p-value and is part of the claim
    gate; a positive point estimate alone is insufficient.
    """
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "p_one_sided": None, "n_strata": 0, "positive_gate": False}
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, array.size))
    observed = float(array.mean())
    null_means = signs @ array / array.size
    p_value = float((1.0 + np.sum(null_means >= observed - 1e-15)) / (draws + 1.0))
    bootstrap = array[rng.integers(0, array.size, size=(draws, array.size))].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "mean": observed,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_one_sided": p_value,
        "n_strata": int(array.size),
        "positive_gate": bool(observed > 0.0 and float(low) > 0.0 and p_value < 0.05),
    }


def _load(path: Path, *, require_external: bool = False) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("record_schema") != "real_checkpoint_selection.v1":
        raise ValueError(f"{path} has incompatible record_schema")
    record = payload.get("record")
    if not isinstance(record, dict):
        raise ValueError(f"{path} has no record")
    # The immutable split/interface declaration lives at the payload level in
    # real_checkpoint_selection.v1.  Copy it into the normalized record before
    # extracting environment and action-distribution factors.
    record["protocol"] = payload.get("protocol", {})
    diagnostics = record.get("diagnostics", {})
    missing = [name for name in ("prediction_mse", "predictor_spectrum", *METRICS) if name not in diagnostics]
    # We keep missing Schur records in the audit ledger so the reason for a
    # gated result is visible, but reject missing core fields immediately.
    core_missing = [name for name in ("prediction_mse", "predictor_spectrum") if name not in diagnostics]
    if core_missing:
        raise ValueError(f"{path} lacks required diagnostics: {core_missing}")
    factors = record.get("model_factors", {})
    missing_factors = [name for name in REQUIRED_FACTORS if factors.get(name) is None and name != "action_distribution"]
    action = _action_distribution(record)
    if action == "missing":
        missing_factors.append("action_distribution")
    if missing_factors:
        raise ValueError(f"{path} lacks fixed factors: {missing_factors}")
    record["_source_path"] = str(path)
    record["_architecture"] = _architecture(record)
    record["_seed"] = _seed(record)
    record["_environment"] = _environment(record)
    record["_stage"] = _stage(record)
    record["_action_distribution"] = action
    record["_latent_dimension"] = int(factors["latent_dimension"])
    record["_spectrum"] = _spectrum(record)
    record["_missing_metrics"] = [
        name for name in METRICS
        if name not in diagnostics or not _finite(diagnostics.get(name))
    ]
    if not _finite(diagnostics["prediction_mse"]):
        raise ValueError(f"{path} has non-finite prediction_mse")
    if not _finite(record.get("heldout", {}).get("planning_loss")):
        raise ValueError(f"{path} lacks finite heldout planning_loss")
    has_external = (
        _finite(record.get("model_factors", {}).get("action_inverse_risk"))
        and _finite(record.get("diagnostics", {}).get("action_inverse_risk"))
        and _finite(record.get("heldout", {}).get("planning_risk"))
        and record.get("protocol", {}).get("independent_external_evidence") is True
    )
    if require_external and not has_external:
        raise ValueError(f"{path} lacks independent action-risk and planning-consequence evidence")
    record["_has_external_evidence"] = bool(has_external)
    return record


def _stratum_key(record: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        record["_environment"],
        record["_stage"],
        record["_latent_dimension"],
        record["_action_distribution"],
    )


def audit(
    records: list[dict[str, Any]],
    *,
    mse_tolerance: float = 0.05,
    spectrum_tolerance: float = 0.05,
    action_risk_tolerance: float = 0.05,
    bootstrap_seed: int = 20261011,
    require_external: bool = False,
) -> dict[str, Any]:
    if len(records) < 2:
        raise ValueError("at least two records are required")
    if mse_tolerance < 0 or spectrum_tolerance < 0 or action_risk_tolerance < 0:
        raise ValueError("matching tolerances must be non-negative")

    groups: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_stratum_key(record)].append(record)

    reports: list[dict[str, Any]] = []
    values_by_metric: dict[str, list[float]] = defaultdict(list)
    regrets_by_metric: dict[str, list[float]] = defaultdict(list)
    observations_by_environment: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    unavailable: dict[str, int] = defaultdict(int)

    for key, candidates in sorted(groups.items()):
        environment, stage, latent_dimension, action_distribution = key
        eligible, compatible_edges = _maximum_compatible_clique(
            candidates,
            mse_tolerance=mse_tolerance,
            spectrum_tolerance=spectrum_tolerance,
            action_risk_tolerance=action_risk_tolerance if require_external else None,
        )
        eligible_records = [candidates[index] for index in eligible]
        missing_schur = any(
            "task_metric_schur_offdiagonal_energy" in item["_missing_metrics"]
            for item in eligible_records
        )
        architectures = sorted({item["_architecture"] for item in eligible_records})
        seeds = sorted({item["_seed"] for item in eligible_records})
        matched = len(eligible_records) >= 2 and len(architectures) >= 2 and len(seeds) >= 2
        pairwise_mse_distances = [
            _symmetric_relative_distance(
                float(left["diagnostics"]["prediction_mse"]),
                float(right["diagnostics"]["prediction_mse"]),
            )
            for left_index, left in enumerate(eligible_records)
            for right in eligible_records[left_index + 1 :]
        ]
        pairwise_spectrum_distances = [
            _spectrum_distance(left["_spectrum"], right["_spectrum"])
            for left_index, left in enumerate(eligible_records)
            for right in eligible_records[left_index + 1 :]
        ]
        oracle_index: int | None = None
        row: dict[str, Any] = {
            "environment": environment,
            "training_stage": stage,
            "latent_dimension": latent_dimension,
            "action_distribution": action_distribution,
            "candidate_count": len(candidates),
            "eligible_count": int(len(eligible_records)),
            "compatible_edge_count": len(compatible_edges),
            "eligible_architectures": architectures,
            "eligible_seeds": seeds,
            "matching_rule": "diversity_prioritized_maximum_pairwise_symmetric_relative_mse_and_spectrum_clique",
            "max_pairwise_mse_distance": max(pairwise_mse_distances, default=0.0),
            "max_pairwise_spectrum_distance": max(pairwise_spectrum_distances, default=0.0),
            "max_pairwise_action_risk_distance": max(
                [
                    _symmetric_relative_distance(
                        float(left["model_factors"].get("action_inverse_risk")),
                        float(right["model_factors"].get("action_inverse_risk")),
                    )
                    for left_index, left in enumerate(eligible_records)
                    for right in eligible_records[left_index + 1 :]
                    if _finite(left["model_factors"].get("action_inverse_risk"))
                    and _finite(right["model_factors"].get("action_inverse_risk"))
                ],
                default=float("nan"),
            ),
            "action_risk_matched": bool(
                all(_finite(item["model_factors"].get("action_inverse_risk")) for item in eligible_records)
            ),
            "missing_task_metric_schur": bool(missing_schur),
            "identifiable": bool(matched and not missing_schur and (not require_external or all(
                item.get("_has_external_evidence", False) for item in eligible_records
            ))),
            "eligible_checkpoints": [item["checkpoint"] for item in eligible_records],
            "selectors": {},
            "rank_correlations": {},
            "reason_if_gated": None,
        }
        if not len(eligible_records) >= 2:
            row["reason_if_gated"] = "fewer_than_two_quality_and_spectrum_matched_candidates"
        elif len(architectures) < 2:
            row["reason_if_gated"] = "only_one_architecture_in_matched_pool"
        elif len(seeds) < 2:
            row["reason_if_gated"] = "only_one_seed_in_matched_pool"
        elif missing_schur:
            row["reason_if_gated"] = "task_metric_schur_missing_from_at_least_one_candidate"
        elif require_external and not all(item.get("_has_external_evidence", False) for item in eligible_records):
            row["reason_if_gated"] = "independent_action_risk_or_planning_consequence_missing"

        if row["identifiable"]:
            outcome_key = "planning_risk" if require_external else "planning_loss"
            risks = np.asarray([float(item["heldout"][outcome_key]) for item in eligible_records])
            oracle_index = int(np.argmin(risks))
            oracle_risk = float(risks[oracle_index])
            # All selectors are lower-is-better and are fixed before reading
            # held-out outcomes.  This is a declared diagnostic direction,
            # not a post-hoc choice based on the observed risk.
            for metric in METRICS:
                metric_values = np.asarray([float(item["diagnostics"][metric]) for item in eligible_records])
                correlation = _spearman(metric_values, risks)
                row["rank_correlations"][metric] = correlation
                observations_by_environment[environment][metric].append(metric_values)
                observations_by_environment[environment][f"{metric}/risk"].append(risks)
                selected = int(np.argmin(metric_values))
                regret = float(risks[selected] - oracle_risk)
                row["selectors"][metric] = {
                    "checkpoint": eligible_records[selected]["checkpoint"],
                    "selected_risk": float(risks[selected]),
                    "regret": regret,
                    "is_oracle": bool(selected == oracle_index),
                }
                values_by_metric[metric].append(float(correlation) if correlation is not None else np.nan)
                regrets_by_metric[metric].append(regret)
            # The primary estimand is fixed before observing planning outcomes:
            # MSE-only versus a rank-averaged MSE + independent action-risk +
            # geometry selector.  All quantities used by both selectors are
            # diagnostic/intervention split measurements.
            mse_values = np.asarray([float(item["diagnostics"]["prediction_mse"]) for item in eligible_records])
            action_values = np.asarray([float(item["diagnostics"].get("action_inverse_risk", np.nan)) for item in eligible_records])
            metric_values = {
                "prediction_mse": mse_values,
                "action_inverse_risk": action_values,
                **{metric: np.asarray([float(item["diagnostics"][metric]) for item in eligible_records]) for metric in METRICS},
            }
            if require_external and not np.isfinite(action_values).all():
                raise ValueError("external audit encountered a non-finite action_inverse_risk")
            rank_matrix = np.stack([_rank(value) for value in metric_values.values()])
            mse_only = int(np.argmin(mse_values))
            full_index = int(np.argmin(rank_matrix.mean(axis=0)))
            geometry_index = int(np.argmin(rank_matrix[2:].mean(axis=0)))
            row["selectors"]["mse_only"] = {
                "checkpoint": eligible_records[mse_only]["checkpoint"],
                "selected_risk": float(risks[mse_only]),
                "regret": float(risks[mse_only] - oracle_risk),
                "is_oracle": bool(mse_only == oracle_index),
            }
            row["selectors"]["full_mse_action_risk_geometry"] = {
                "checkpoint": eligible_records[full_index]["checkpoint"],
                "selected_risk": float(risks[full_index]),
                "regret": float(risks[full_index] - oracle_risk),
                "is_oracle": bool(full_index == oracle_index),
            }
            row["selectors"]["geometry_composite"] = {
                "checkpoint": eligible_records[geometry_index]["checkpoint"],
                "selected_risk": float(risks[geometry_index]),
                "regret": float(risks[geometry_index] - oracle_risk),
                "is_oracle": bool(geometry_index == oracle_index),
            }
            delta = float(risks[mse_only] - risks[full_index])
            row["delta_risk_mse_minus_full"] = delta
            regrets_by_metric["mse_only"].append(float(risks[mse_only] - oracle_risk))
            regrets_by_metric["full_mse_action_risk_geometry"].append(float(risks[full_index] - oracle_risk))
            regrets_by_metric["geometry_composite"].append(float(risks[geometry_index] - oracle_risk))
            values_by_metric["delta_risk_mse_minus_full"].append(delta)
            mse_selected = int(np.argmin(np.asarray([float(item["diagnostics"]["prediction_mse"]) for item in eligible_records])))
            random_regret = float(risks.mean() - oracle_risk)
            row["selectors"]["prediction_mse_tie_break"] = {
                "checkpoint": eligible_records[mse_selected]["checkpoint"],
                "selected_risk": float(risks[mse_selected]),
                "regret": float(risks[mse_selected] - oracle_risk),
                "is_oracle": bool(mse_selected == oracle_index),
            }
            row["selectors"]["random_expected"] = {
                "selected_risk": float(risks.mean()),
                "regret": random_regret,
                "is_oracle": None,
            }
            regrets_by_metric["prediction_mse_tie_break"].append(float(risks[mse_selected] - oracle_risk))
            regrets_by_metric["random_expected"].append(random_regret)
        else:
            for metric in (*METRICS, "prediction_mse_tie_break", "random_expected", "mse_only", "full_mse_action_risk_geometry", "geometry_composite"):
                unavailable[metric] += 1
        reports.append(row)

    environments = sorted({record["_environment"] for record in records})
    architectures = sorted({record["_architecture"] for record in records})
    seeds = sorted({record["_seed"] for record in records})
    identifiable = [row for row in reports if row["identifiable"]]
    identifiable_environments = sorted({row["environment"] for row in identifiable})
    summary = {
        "identifiable_strata": len(identifiable),
        "identifiable_environments": identifiable_environments,
        "identifiable_environment_count": len(identifiable_environments),
        "total_strata": len(reports),
        "environment_count": len(environments),
        "architecture_count": len(architectures),
        "seed_count": len(seeds),
        "selection_regret": {
            name: _bootstrap([value for value in values if np.isfinite(value)], bootstrap_seed + index)
            for index, (name, values) in enumerate(sorted(regrets_by_metric.items()))
        },
        "rank_correlation": {
            name: _bootstrap([value for value in values if np.isfinite(value)], bootstrap_seed + 100 + index)
            for index, (name, values) in enumerate(sorted(values_by_metric.items()))
        },
        "leave_one_environment_out_rank_correlation": {},
        "unavailable_strata": dict(sorted(unavailable.items())),
    }
    delta_values = [float(value) for value in values_by_metric["delta_risk_mse_minus_full"] if np.isfinite(value)]
    summary["primary_delta_risk_mse_minus_full"] = _positive_delta_test(delta_values, bootstrap_seed + 500)
    # Leave-one-environment-out is computed only from identifiable strata and
    # is omitted when removing an environment leaves no independent outcome
    # observations.  It is a generalization audit, not a tuning loop.
    for metric in METRICS:
        loo_values: list[float] = []
        for heldout_environment in sorted(observations_by_environment):
            metric_arrays = [
                array
                for environment, by_metric in observations_by_environment.items()
                if environment != heldout_environment
                for array in by_metric.get(metric, [])
            ]
            risk_arrays = [
                array
                for environment, by_metric in observations_by_environment.items()
                if environment != heldout_environment
                for array in by_metric.get(f"{metric}/risk", [])
            ]
            if metric_arrays and risk_arrays:
                value = _spearman(np.concatenate(metric_arrays), np.concatenate(risk_arrays))
                if value is not None:
                    loo_values.append(value)
        summary["leave_one_environment_out_rank_correlation"][metric] = _bootstrap(
            loo_values, bootstrap_seed + 200 + len(summary["leave_one_environment_out_rank_correlation"])
        )
    positive_gate = summary["primary_delta_risk_mse_minus_full"]["positive_gate"]
    if len(identifiable) == 0 and any(row["missing_task_metric_schur"] and row["eligible_count"] >= 2 for row in reports):
        status = "gated_missing_schur_metric"
    elif len(identifiable) == 0:
        status = "gated_no_identifiable_matched_stratum"
    elif len(environments) < 2:
        status = "gated_single_environment"
    elif len(identifiable_environments) < 2:
        status = "gated_insufficient_environment_replication"
    elif len(architectures) < 2:
        status = "gated_single_architecture"
    elif any(row["missing_task_metric_schur"] for row in identifiable):
        status = "gated_missing_schur_metric"
    else:
        status = "conditional_transfer_audit_complete"
    summary["status"] = status
    summary["claim_ready_for_positive_delta"] = bool(
        positive_gate and len(identifiable_environments) >= 2 and len(architectures) >= 2 and len(seeds) >= 2
    )
    if status == "conditional_transfer_audit_complete" and not summary["claim_ready_for_positive_delta"]:
        summary["status"] = "gated_positive_delta_not_established"
    return {
        "protocol": {
            "estimand": "conditional independent planning-risk difference after matching MSE, latent dimension, predictor spectrum, training stage, action distribution, and (strict mode) action inverse risk",
            "matching": {
                "prediction_mse": f"symmetric relative pairwise distance <= {mse_tolerance}",
                "latent_dimension": "exact",
                "predictor_spectrum": f"normalized max eigenvalue distance <= {spectrum_tolerance}",
                "training_stage": "exact",
                "action_distribution": "exact declaration",
                "action_inverse_risk": f"symmetric relative pairwise distance <= {action_risk_tolerance}" if require_external else "not required in legacy mode",
            },
            "metric_fit_split": "diagnostic/train split only as recorded by source protocol",
            "outcome": "independent held-out planning risk" if require_external else "held-out outcome recorded by source protocol",
            "predeclared_metrics": list(METRICS),
            "selector_direction": "lower is better for all four diagnostics",
            "replication_unit": "environment stratum; not episode-level pseudoreplication",
            "negative_result_policy": "unmatched, missing-Schur, and negative strata remain visible and cannot be overwritten",
            "require_independent_external_evidence": bool(require_external),
            "primary_estimand": "Delta R = R(MSE-only) - R(MSE+action-risk+geometry)",
        },
        "inventory": {
            "environments": environments,
            "architectures": architectures,
            "seeds": seeds,
        },
        "strata": reports,
        "summary": summary,
        "claim_boundary": (
            "A conditional transfer-risk claim is eligible only after the primary four-metric benchmark has identifiable matched strata replicated across at least two architectures, two seeds, and two environments. "
            "This audit cannot establish a universal model-selection criterion or a commutator certificate from unmatched learned checkpoints."
        ),
    }


def markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Matched learned transfer-risk benchmark",
        "",
        "The audit fixes prediction quality, latent dimension, predictor spectrum, training stage, and action distribution before evaluating task/action geometry.",
        "",
        f"**Status:** `{summary['status']}`  ",
        f"**Identifiable strata:** {summary['identifiable_strata']} / {summary['total_strata']}  ",
        f"**Identifiable environments:** {summary['identifiable_environment_count']} / {summary['environment_count']}  ",
        f"**Environments / architectures / seeds:** {summary['environment_count']} / {summary['architecture_count']} / {summary['seed_count']}",
        "",
        f"**Primary Delta R = R(MSE-only) - R(full):** `{summary['primary_delta_risk_mse_minus_full']}`  ",
        f"**Positive-Delta claim ready:** `{summary['claim_ready_for_positive_delta']}`",
        "",
        "| Diagnostic | Rank correlation mean [95% CI] | Selection regret mean [95% CI] |",
        "|---|---:|---:|",
    ]
    for name in (*METRICS, "prediction_mse_tie_break", "random_expected"):
        corr = summary["rank_correlation"].get(name, {})
        regret = summary["selection_regret"].get(name, {})
        corr_text = "NA" if corr.get("mean") is None else f"{corr['mean']:.6g} [{corr['ci95_low']:.6g}, {corr['ci95_high']:.6g}]"
        regret_text = "NA" if regret.get("mean") is None else f"{regret['mean']:.6g} [{regret['ci95_low']:.6g}, {regret['ci95_high']:.6g}]"
        lines.append(f"| {name} | {corr_text} | {regret_text} |")
    lines.extend(["", "## Leave-one-environment-out rank correlation", "", "| Diagnostic | Mean [95% CI] |", "|---|---:|"])
    for name in METRICS:
        value = summary["leave_one_environment_out_rank_correlation"].get(name, {})
        text = "NA" if value.get("mean") is None else f"{value['mean']:.6g} [{value['ci95_low']:.6g}, {value['ci95_high']:.6g}]"
        lines.append(f"| {name} | {text} |")
    lines.extend(["", "## Gating reasons", "", "| Environment | Stage | Eligible | Architectures | Seeds | Reason |", "|---|---:|---:|---|---|---|"])
    for row in report["strata"]:
        lines.append(
            f"| {row['environment']} | {row['training_stage']} | {row['eligible_count']} | "
            f"{', '.join(row['eligible_architectures']) or '—'} | {', '.join(row['eligible_seeds']) or '—'} | "
            f"{row['reason_if_gated'] or 'identifiable'} |"
        )
    lines.extend(["", "## Claim boundary", "", report["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="append", type=Path)
    parser.add_argument("--record-root", action="append", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mse-tolerance", type=float, default=0.05)
    parser.add_argument("--spectrum-tolerance", type=float, default=0.05)
    parser.add_argument("--action-risk-tolerance", type=float, default=0.05)
    parser.add_argument("--require-independent-external", action="store_true")
    args = parser.parse_args()
    paths = list(args.record or [])
    if args.record_root:
        # Accept both the historical candidate_000/checkpoint_record.json
        # layout and deeper experiment bundles.  A non-recursive fixed-depth
        # glob silently produced an empty audit for valid one-level records.
        for record_root in args.record_root:
            paths.extend(sorted(record_root.rglob("checkpoint_record.json")))
    if not paths:
        parser.error("provide --record or --record-root")
    report = audit(
        [_load(path, require_external=args.require_independent_external) for path in paths],
        mse_tolerance=args.mse_tolerance,
        spectrum_tolerance=args.spectrum_tolerance,
        action_risk_tolerance=args.action_risk_tolerance,
        require_external=args.require_independent_external,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "matched_learned_transfer_benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "matched_learned_transfer_benchmark.md").write_text(markdown(report))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
