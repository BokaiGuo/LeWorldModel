"""Training-only checkpoint selection protocol for real learned world models.

This module deliberately contains no learned-model code.  A runner first
exports one immutable record per (checkpoint, environment, seed, horizon):
all diagnostics are computed on the diagnostic split, and the planning loss is
computed only on held-out episodes.  This module then applies pre-declared
selectors and reports their held-out regret.  Keeping the two stages separate
makes it impossible to tune a selector on the planning score by accident.

The protocol is a test of model-selection utility, not a claim that any single
matrix statistic is a universal transfer certificate.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


REQUIRED_DIAGNOSTICS = (
    "prediction_mse",
    "alignment",
    "normalized_commutator",
    "spectral_radius",
    "latent_variance",
    "schur_offdiagonal_energy",
    "departure_from_normality",
    "transient_amplification",
    "pseudospectral_sensitivity",
)

# New records may additionally carry a fixed external task-space metric.  The
# fields are optional so that the original negative benchmark remains
# readable and reproducible; a record missing them cannot silently enter the
# task-metric selector comparison.
OPTIONAL_TASK_DIAGNOSTICS = (
    "task_metric_alignment",
    "task_metric_commutator",
    "task_metric_departure_from_normality",
    "task_metric_transient_amplification",
    "task_metric_spectral_radius",
)


@dataclass(frozen=True)
class Candidate:
    group: str
    checkpoint: str
    diagnostics: dict[str, float]
    heldout_planning_loss: float


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def candidate_from_record(record: dict[str, Any]) -> Candidate:
    try:
        diagnostics = {name: _finite(record["diagnostics"][name], name) for name in REQUIRED_DIAGNOSTICS}
        diagnostics.update({
            name: _finite(record["diagnostics"][name], name)
            for name in OPTIONAL_TASK_DIAGNOSTICS
            if name in record.get("diagnostics", {})
        })
        return Candidate(
            group=str(record["group"]),
            checkpoint=str(record["checkpoint"]),
            diagnostics=diagnostics,
            heldout_planning_loss=_finite(record["heldout"]["planning_loss"], "planning_loss"),
        )
    except KeyError as exc:
        raise ValueError(f"incomplete checkpoint record missing {exc.args[0]!r}") from exc


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling; smaller values are better."""
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return result


def _selector_indices(
    candidates: list[Candidate], random_seed: int, mse_relative_tolerance: float,
) -> tuple[dict[str, int | float | None], np.ndarray]:
    """Pre-declared selectors using diagnostic-split values only.

    ``geometry_composite`` assigns equal rank weight to alignment, normalized
    commutator, Schur off-diagonal energy, departure from normality, finite-horizon
    amplification.  It intentionally excludes held-out planning and prediction
    MSE, so its added value over MSE-only is separately identifiable.
    """
    values = {name: np.asarray([item.diagnostics[name] for item in candidates]) for name in REQUIRED_DIAGNOSTICS}
    if all(name in item.diagnostics for item in candidates for name in OPTIONAL_TASK_DIAGNOSTICS):
        values.update({name: np.asarray([item.diagnostics[name] for item in candidates]) for name in OPTIONAL_TASK_DIAGNOSTICS})
    geometry = np.mean(
        np.stack([
            _ranks(values["alignment"]),
            _ranks(values["normalized_commutator"]),
            _ranks(values["schur_offdiagonal_energy"]),
            _ranks(values["departure_from_normality"]),
            _ranks(values["transient_amplification"]),
            _ranks(values["pseudospectral_sensitivity"]),
        ]),
        axis=0,
    )
    # Spectrum-only is deliberately blind to eigenvectors and non-normality.
    # A stable finite-horizon predictor has spectral radius nearest to one in
    # this protocol; this fixed rule is declared before held-out evaluation.
    spectrum_distance = np.abs(values["spectral_radius"] - 1.0)
    task_geometry = None
    if all(name in values for name in OPTIONAL_TASK_DIAGNOSTICS):
        # This is a separate, externally specified task-space geometry.  It is
        # intentionally not pooled with the legacy Euclidean diagnostics: the
        # experiment asks whether a declared non-degenerate metric adds value.
        task_geometry = np.mean(
            np.stack([
                _ranks(values["task_metric_alignment"]),
                _ranks(values["task_metric_commutator"]),
                _ranks(values["task_metric_departure_from_normality"]),
                _ranks(values["task_metric_transient_amplification"]),
                _ranks(values["task_metric_spectral_radius"]),
            ]),
            axis=0,
        )
    if mse_relative_tolerance < 0.0:
        raise ValueError("mse_relative_tolerance must be nonnegative")
    mse_floor = float(values["prediction_mse"].min())
    mse_matched = np.flatnonzero(values["prediction_mse"] <= mse_floor * (1.0 + mse_relative_tolerance))
    selectors: dict[str, int | float | None] = {
        "alignment_only": int(np.argmin(values["alignment"])),
        "commutator_only": int(np.argmin(values["normalized_commutator"])),
        "geometry_composite": int(np.argmin(geometry)),
        "prediction_mse_only": int(np.argmin(values["prediction_mse"])),
        "spectrum_only": int(np.argmin(spectrum_distance)),
        "variance_only": int(np.argmax(values["latent_variance"])),
        "random_expected": float(np.mean(np.arange(len(candidates), dtype=np.float64))),
    }
    if all(name in values for name in OPTIONAL_TASK_DIAGNOSTICS):
        selectors["task_metric_alignment_only"] = int(np.argmin(values["task_metric_alignment"]))
        selectors["task_metric_geometry_composite"] = int(np.argmin(task_geometry))
    # A singleton quality filter is a quality-selection result, not evidence
    # that geometry resolved a tie.  Keep it explicitly non-identifiable.
    if mse_matched.size >= 2:
        selectors.update({
            "mse_filtered_alignment": int(mse_matched[np.argmin(values["alignment"][mse_matched])]),
            "mse_filtered_commutator": int(mse_matched[np.argmin(values["normalized_commutator"][mse_matched])]),
            "mse_filtered_geometry_composite": int(mse_matched[np.argmin(geometry[mse_matched])]),
        })
        if all(name in values for name in OPTIONAL_TASK_DIAGNOSTICS):
            selectors["mse_filtered_task_metric_alignment"] = int(
                mse_matched[np.argmin(values["task_metric_alignment"][mse_matched])]
            )
            selectors["mse_filtered_task_metric_geometry_composite"] = int(
                mse_matched[np.argmin(task_geometry[mse_matched])]
            )
    else:
        selectors.update({
            "mse_filtered_alignment": None,
            "mse_filtered_commutator": None,
            "mse_filtered_geometry_composite": None,
        })
        if all(name in values for name in OPTIONAL_TASK_DIAGNOSTICS):
            selectors["mse_filtered_task_metric_alignment"] = None
            selectors["mse_filtered_task_metric_geometry_composite"] = None
    return selectors, mse_matched


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    return float(np.corrcoef(_ranks(left), _ranks(right))[0, 1])


def _kendall(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    concordant = discordant = 0
    for row in range(len(left)):
        for column in range(row + 1, len(left)):
            product = (left[row] - left[column]) * (right[row] - right[column])
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    denominator = concordant + discordant
    return None if denominator == 0 else float((concordant - discordant) / denominator)


def _mean_ci(values: Iterable[float], bootstrap_seed: int, repetitions: int = 4000) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty set")
    rng = np.random.default_rng(bootstrap_seed)
    means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        means[index] = float(rng.choice(array, size=array.size, replace=True).mean())
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n_groups": int(array.size),
    }


def _epoch_from_checkpoint(label: str) -> int | None:
    matched = re.search(r"epoch[_/](\d+)", label)
    return None if matched is None else int(matched.group(1))


def _conditional_regression(candidates: list[Candidate], losses: np.ndarray) -> dict[str, Any]:
    """Descriptive increment after quality/stage control, never a causal test."""
    stages = [_epoch_from_checkpoint(item.checkpoint) for item in candidates]
    if any(value is None for value in stages):
        return {"status": "not_identifiable", "reason": "checkpoint labels do not encode training stage"}
    if len(candidates) < 6:
        return {"status": "not_identifiable", "reason": "fewer than six candidates for five regression coefficients", "n": len(candidates)}
    mse = np.asarray([item.diagnostics["prediction_mse"] for item in candidates], dtype=np.float64)
    variance = np.asarray([item.diagnostics["latent_variance"] for item in candidates], dtype=np.float64)
    alignment = np.asarray([item.diagnostics["alignment"] for item in candidates], dtype=np.float64)
    stage = np.asarray(stages, dtype=np.float64)
    def standardized(value: np.ndarray) -> np.ndarray:
        scale = float(value.std())
        if scale == 0.0:
            raise ValueError("a controlled regression covariate is constant")
        return (value - value.mean()) / scale
    try:
        base = np.column_stack([np.ones(len(candidates)), standardized(mse), standardized(variance), standardized(stage)])
        full = np.column_stack([base, standardized(alignment)])
        if np.linalg.matrix_rank(full) < full.shape[1]:
            return {"status": "not_identifiable", "reason": "rank-deficient MSE/variance/stage/alignment design", "n": len(candidates)}
        beta_base = np.linalg.lstsq(base, losses, rcond=None)[0]
        beta_full = np.linalg.lstsq(full, losses, rcond=None)[0]
        sst = float(np.square(losses - losses.mean()).sum())
        return {
            "status": "descriptive_only",
            "n": len(candidates),
            "alignment_coefficient_standardized": float(beta_full[-1]),
            "base_r2": float(1.0 - np.square(losses - base @ beta_base).sum() / max(sst, 1e-12)),
            "full_r2": float(1.0 - np.square(losses - full @ beta_full).sum() / max(sst, 1e-12)),
            "incremental_r2": float(
                np.square(losses - base @ beta_base).sum() / max(sst, 1e-12)
                - np.square(losses - full @ beta_full).sum() / max(sst, 1e-12)
            ),
        }
    except ValueError as exc:
        return {"status": "not_identifiable", "reason": str(exc), "n": len(candidates)}


def _pooled_controlled_alignment_increment(by_group: dict[str, list[Candidate]]) -> dict[str, Any]:
    """Descriptive alignment increment after within-group MSE/variance/stage control.

    Six snapshots per group leave only one residual degree of freedom for the
    requested four-covariate regression.  The pre-registered analysis therefore
    pools only after removing a separate intercept from every fixed
    architecture-and-seed group; it reports no p-value or causal claim.
    """
    if len(by_group) < 3:
        return {"status": "not_identifiable", "reason": "fewer than three fixed candidate groups"}
    columns: dict[str, list[np.ndarray]] = {name: [] for name in ("loss", "mse", "variance", "stage", "alignment")}
    usable_groups = 0
    for candidates in by_group.values():
        stages = np.asarray([_epoch_from_checkpoint(item.checkpoint) for item in candidates], dtype=object)
        if len(candidates) < 5 or any(value is None for value in stages):
            continue
        raw = {
            "loss": np.asarray([item.heldout_planning_loss for item in candidates], dtype=np.float64),
            "mse": np.asarray([item.diagnostics["prediction_mse"] for item in candidates], dtype=np.float64),
            "variance": np.asarray([item.diagnostics["latent_variance"] for item in candidates], dtype=np.float64),
            "stage": np.asarray(stages, dtype=np.float64),
            "alignment": np.asarray([item.diagnostics["alignment"] for item in candidates], dtype=np.float64),
        }
        if any(np.ptp(value) == 0.0 for value in raw.values()):
            continue
        for name, value in raw.items():
            columns[name].append((value - value.mean()) / value.std())
        usable_groups += 1
    if usable_groups < 3:
        return {"status": "not_identifiable", "reason": "fewer than three non-degenerate fixed groups"}
    merged = {name: np.concatenate(value) for name, value in columns.items()}
    base = np.column_stack([merged["mse"], merged["variance"], merged["stage"]])
    full = np.column_stack([base, merged["alignment"]])
    if np.linalg.matrix_rank(full) < full.shape[1]:
        return {"status": "not_identifiable", "reason": "rank-deficient pooled controlled design", "n_groups": usable_groups}
    beta_base = np.linalg.lstsq(base, merged["loss"], rcond=None)[0]
    beta_full = np.linalg.lstsq(full, merged["loss"], rcond=None)[0]
    sst = float(np.square(merged["loss"]).sum())
    return {
        "status": "descriptive_only",
        "n_groups": usable_groups,
        "n_candidates": int(merged["loss"].size),
        "alignment_coefficient_standardized": float(beta_full[-1]),
        "base_r2_within_group": float(1.0 - np.square(merged["loss"] - base @ beta_base).sum() / max(sst, 1e-12)),
        "full_r2_within_group": float(1.0 - np.square(merged["loss"] - full @ beta_full).sum() / max(sst, 1e-12)),
        "incremental_r2_within_group": float(
            np.square(merged["loss"] - base @ beta_base).sum() / max(sst, 1e-12)
            - np.square(merged["loss"] - full @ beta_full).sum() / max(sst, 1e-12)
        ),
        "interpretation": "descriptive pooled fixed-group regression; not a causal test or a universal certificate",
    }


def evaluate_records(
    records: Iterable[dict[str, Any]], bootstrap_seed: int = 20261009, mse_relative_tolerance: float = 0.05,
) -> dict[str, Any]:
    candidates = [candidate_from_record(record) for record in records]
    by_group: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_group[candidate.group].append(candidate)
    if not by_group:
        raise ValueError("no checkpoint records supplied")

    group_reports: list[dict[str, Any]] = []
    regret_by_selector: dict[str, list[float]] = defaultdict(list)
    unavailable_by_selector: dict[str, int] = defaultdict(int)
    corr_by_metric: dict[str, list[float]] = defaultdict(list)
    kendall_by_metric: dict[str, list[float]] = defaultdict(list)
    for group_index, (group, group_candidates) in enumerate(sorted(by_group.items())):
        if len(group_candidates) < 2:
            raise ValueError(f"group {group!r} needs at least two checkpoint candidates")
        losses = np.asarray([item.heldout_planning_loss for item in group_candidates])
        oracle = int(np.argmin(losses))
        selected, mse_matched = _selector_indices(group_candidates, bootstrap_seed + group_index, mse_relative_tolerance)
        selector_regret: dict[str, float | None] = {}
        for name, index in selected.items():
            if index is None:
                selector_regret[name] = None
                unavailable_by_selector[name] += 1
                continue
            if name == "random_expected":
                value = float(losses.mean() - losses[oracle])
            else:
                value = float(losses[int(index)] - losses[oracle])
            selector_regret[name] = value
            regret_by_selector[name].append(value)
        correlations = {}
        present_metrics = list(REQUIRED_DIAGNOSTICS)
        if all(name in item.diagnostics for item in group_candidates for name in OPTIONAL_TASK_DIAGNOSTICS):
            present_metrics.extend(OPTIONAL_TASK_DIAGNOSTICS)
        for metric in present_metrics:
            corr = _spearman(
                np.asarray([item.diagnostics[metric] for item in group_candidates]), losses,
            )
            correlations[metric] = corr
            if corr is not None:
                corr_by_metric[metric].append(corr)
            tau = _kendall(
                np.asarray([item.diagnostics[metric] for item in group_candidates]), losses,
            )
            correlations[f"{metric}_kendall"] = tau
            if tau is not None:
                kendall_by_metric[metric].append(tau)
        matched_pairs = []
        for left in range(len(group_candidates)):
            for right in range(left + 1, len(group_candidates)):
                mse_left = group_candidates[left].diagnostics["prediction_mse"]
                mse_right = group_candidates[right].diagnostics["prediction_mse"]
                relative_gap = abs(mse_left - mse_right) / max(min(mse_left, mse_right), 1e-12)
                if relative_gap <= mse_relative_tolerance:
                    matched_pairs.append({
                        "left": group_candidates[left].checkpoint,
                        "right": group_candidates[right].checkpoint,
                        "relative_prediction_mse_gap": float(relative_gap),
                        "planning_loss_left_minus_right": float(losses[left] - losses[right]),
                        "alignment_left_minus_right": float(group_candidates[left].diagnostics["alignment"] - group_candidates[right].diagnostics["alignment"]),
                        "commutator_left_minus_right": float(group_candidates[left].diagnostics["normalized_commutator"] - group_candidates[right].diagnostics["normalized_commutator"]),
                    })
                    if all(name in group_candidates[left].diagnostics and name in group_candidates[right].diagnostics for name in OPTIONAL_TASK_DIAGNOSTICS):
                        matched_pairs[-1]["task_metric_alignment_left_minus_right"] = float(
                            group_candidates[left].diagnostics["task_metric_alignment"]
                            - group_candidates[right].diagnostics["task_metric_alignment"]
                        )
                        matched_pairs[-1]["task_metric_commutator_left_minus_right"] = float(
                            group_candidates[left].diagnostics["task_metric_commutator"]
                            - group_candidates[right].diagnostics["task_metric_commutator"]
                        )
                        matched_pairs[-1]["task_metric_transient_left_minus_right"] = float(
                            group_candidates[left].diagnostics["task_metric_transient_amplification"]
                            - group_candidates[right].diagnostics["task_metric_transient_amplification"]
                        )
        group_reports.append({
            "group": group,
            "oracle_checkpoint": group_candidates[oracle].checkpoint,
            "oracle_heldout_planning_loss": float(losses[oracle]),
            "candidate_count": len(group_candidates),
            "mse_quality_filter": {
                "relative_tolerance": mse_relative_tolerance,
                "candidate_count": int(mse_matched.size),
                "candidates": [group_candidates[int(index)].checkpoint for index in mse_matched],
                "geometry_tie_break_identifiable": bool(mse_matched.size >= 2),
            },
            "selector_regret": selector_regret,
            "diagnostic_planning_spearman": correlations,
            "controlled_alignment_increment": _conditional_regression(group_candidates, losses),
            "prediction_mse_matched_pairs": matched_pairs,
            "candidates": [
                {
                    "checkpoint": item.checkpoint,
                    "diagnostics": item.diagnostics,
                    "heldout_planning_loss": item.heldout_planning_loss,
                }
                for item in group_candidates
            ],
        })
    selector_protocol = {
        "alignment_only": "minimize covariance--predictor commutator alignment",
        "geometry_composite": "equal-rank mean of alignment, Schur off-diagonal energy, departure from normality, transient amplification",
        "commutator_only": "minimize the normalized covariance--predictor commutator",
        "prediction_mse_only": "minimize diagnostic-split one-step prediction MSE",
        "spectrum_only": "spectral radius nearest one",
        "variance_only": "maximize diagnostic-split latent variance",
        "random_expected": "uniform expected regret over available checkpoints",
        "mse_filtered_alignment": "first retain checkpoints within the declared MSE tolerance, then minimize alignment",
        "mse_filtered_commutator": "first retain checkpoints within the declared MSE tolerance, then minimize normalized commutator",
        "mse_filtered_geometry_composite": "first retain checkpoints within the declared MSE tolerance, then minimize geometry composite",
    }
    if any("task_metric_alignment" in item.diagnostics for item in candidates):
        selector_protocol.update({
            "task_metric_alignment_only": "minimize alignment in the pre-declared external PushT task-space metric",
            "task_metric_geometry_composite": "equal-rank mean of task-metric alignment, commutator, Schur/non-normality, transient amplification, and spectral-radius diagnostics",
            "mse_filtered_task_metric_alignment": "first retain checkpoints within the declared MSE tolerance, then minimize task-space alignment",
            "mse_filtered_task_metric_geometry_composite": "first retain checkpoints within the declared MSE tolerance, then minimize the task-space geometry composite",
        })
    return {
        "protocol": {
            "selection_data": "diagnostic split only",
            "outcome_data": "held-out planning episodes only",
            "selectors": selector_protocol,
            "coordinate_gauge_requirement": "each record must declare one fixed, comparable representation coordinate system",
            "mse_relative_tolerance": mse_relative_tolerance,
        },
        "groups": group_reports,
        "summary": {
            "selection_regret": {
                name: _mean_ci(values, bootstrap_seed + 10_000 + index)
                for index, (name, values) in enumerate(sorted(regret_by_selector.items()))
            },
            "diagnostic_planning_spearman": {
                name: _mean_ci(values, bootstrap_seed + 20_000 + index)
                for index, (name, values) in enumerate(sorted(corr_by_metric.items()))
            },
            "diagnostic_planning_kendall": {
                name: _mean_ci(values, bootstrap_seed + 30_000 + index)
                for index, (name, values) in enumerate(sorted(kendall_by_metric.items()))
            },
            "selector_unavailable_groups": dict(sorted(unavailable_by_selector.items())),
            "pooled_controlled_alignment_increment": _pooled_controlled_alignment_increment(by_group),
        },
        "claim_boundary": (
            "A selector is useful only when its regret is lower than each declared baseline "
            "with a confidence interval excluding zero across independently trained checkpoints and environments. "
            "This protocol cannot establish a universal certificate from one environment or one checkpoint family."
        ),
    }
