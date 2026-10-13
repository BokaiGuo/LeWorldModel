"""Held-out covariance--predictor diagnostic on frozen learned checkpoints.

This benchmark is the learned-checkpoint counterpart of the controlled linear
diagnostic.  It consumes per-episode target/predicted vectors exported by a
released JEPA-WM or DINO-WM checkpoint and evaluates a *checkpoint-derived
operator family*:

* fit one low-rank linear predictor on the split-train vectors;
* generate orthogonal conjugates of that fitted operator, preserving its
  eigenvalue multiset exactly;
* use train-only covariance--predictor alignment for selection;
* use a validation split to choose a one-step-error-matched pair; and
* evaluate one-step error and recurrent multi-step transfer on a held-out
  test split.

The conjugated matrices are diagnostic probes, not independently trained
neural checkpoints.  The protocol therefore supports a learned-latent,
frozen-checkpoint diagnostic claim, not a claim that alignment alone improves
end-to-end neural training or online planning.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Config:
    fit_horizon: int = 1
    downstream_horizon: int = 4
    rank: int = 16
    ridge: float = 1e-2
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    candidate_count: int = 17
    prediction_tolerance: float = 0.05
    split_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    seed: int = 20261008


def _finite(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        if not np.isfinite(float(value)):
            raise FloatingPointError(f"non-finite value: {value}")


def _fit_basis(x_train: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    centered = x_train - mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    effective_rank = min(int(rank), vh.shape[0], max(1, x_train.shape[0] - 1))
    if effective_rank < 2:
        raise ValueError("at least two train rows are required for the operator probe")
    return mean, vh[:effective_rank]


def _project(values: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - mean) @ basis.T


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    gram = x.T @ x / max(x.shape[0], 1)
    scale = max(float(np.trace(gram)) / max(x.shape[1], 1), 1e-12)
    return np.linalg.solve(
        gram + float(ridge) * scale * np.eye(x.shape[1]),
        x.T @ y / max(x.shape[0], 1),
    )


def _orthogonal(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs[None, :]


def _candidate_family(anchor: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    if count < 2:
        raise ValueError("candidate_count must be at least two")
    result = [np.asarray(anchor, dtype=np.float64)]
    for index in range(1, int(count)):
        q = _orthogonal(anchor.shape[0], seed + index)
        result.append(q @ anchor @ q.T)
    return result


def _alignment(covariance: np.ndarray, operator: np.ndarray) -> float:
    commutator = covariance @ operator - operator @ covariance
    denominator = max(
        float(np.linalg.norm(covariance, "fro") * np.linalg.norm(operator, "fro")),
        1e-12,
    )
    return float(np.linalg.norm(commutator, "fro") / denominator)


def _geometry(covariance: np.ndarray, operator: np.ndarray, horizon: int) -> dict[str, float]:
    """Return non-normal diagnostics for one candidate operator.

    These are computed from the same train-only covariance and the candidate
    operator.  They are reported separately because no single scalar is a
    universal certificate for a non-normal learned predictor.
    """
    triangular, _ = __import__("scipy.linalg", fromlist=["schur"]).schur(operator, output="real")
    norm = max(float(np.linalg.norm(operator, "fro")), 1e-12)
    departure = np.linalg.norm(operator.T @ operator - operator @ operator.T, "fro") / max(norm * norm, 1e-12)
    gains = [float(np.linalg.svd(np.linalg.matrix_power(operator, h), compute_uv=False)[0]) for h in range(1, int(horizon) + 1)]
    return {
        "alignment": _alignment(covariance, operator),
        "schur_offdiagonal_energy": float(np.linalg.norm(np.triu(triangular, k=1), "fro") / norm),
        "departure_from_normality": float(departure),
        "transient_amplification": float(max(gains)),
    }


def _mse(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(predicted) - np.asarray(target))))


def _evaluate(
    x: np.ndarray,
    one_step_target: np.ndarray,
    downstream_target: np.ndarray,
    operator: np.ndarray,
    horizon: int,
) -> dict[str, float]:
    one_step_prediction = x @ operator
    multi_step_prediction = x @ np.linalg.matrix_power(operator, int(horizon))
    return {
        "one_step_mse": _mse(one_step_prediction, one_step_target),
        "downstream_transfer_mse": _mse(multi_step_prediction, downstream_target),
    }


def _split_indices(n: int, config: Config, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < config.train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if config.train_fraction + config.validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be below one")
    order = np.random.default_rng(seed).permutation(n)
    train_end = max(2, int(round(n * config.train_fraction)))
    validation_end = max(train_end + 1, int(round(n * (config.train_fraction + config.validation_fraction))))
    if validation_end >= n:
        raise ValueError(f"need at least one test row; got n={n}")
    return np.sort(order[:train_end]), np.sort(order[train_end:validation_end]), np.sort(order[validation_end:])


def _eigenvalue_match(anchor: np.ndarray, candidates: list[np.ndarray]) -> float:
    reference = np.sort_complex(np.linalg.eigvals(anchor))
    return max(float(np.max(np.abs(np.sort_complex(np.linalg.eigvals(item)) - reference))) for item in candidates)


def _rank_correlation(alignment: np.ndarray, transfer: np.ndarray) -> float:
    if alignment.size < 2 or np.std(alignment) == 0.0 or np.std(transfer) == 0.0:
        return float("nan")
    rank_a = np.argsort(np.argsort(alignment, kind="mergesort"), kind="mergesort")
    rank_t = np.argsort(np.argsort(transfer, kind="mergesort"), kind="mergesort")
    return float(np.corrcoef(rank_a, rank_t)[0, 1])


def _load_vectors(path: Path, fit_horizon: int, downstream_horizon: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    vectors = payload.get("operator_fit_vectors")
    if not isinstance(vectors, dict) or "target" not in vectors or "predicted" not in vectors:
        raise ValueError(f"{path} lacks operator_fit_vectors")
    key_fit, key_down = str(fit_horizon), str(downstream_horizon)
    try:
        fit_target = np.asarray(vectors["target"][key_fit], dtype=np.float64)
        fit_predicted = np.asarray(vectors["predicted"][key_fit], dtype=np.float64)
        downstream_target = np.asarray(vectors["target"][key_down], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"{path} lacks requested horizon {exc.args[0]}") from exc
    if fit_target.ndim != 2 or fit_target.shape != fit_predicted.shape:
        raise ValueError(f"invalid fit vectors in {path}: {fit_target.shape}, {fit_predicted.shape}")
    if downstream_target.shape != fit_target.shape:
        raise ValueError(
            f"fit/downstream rows or dimensions differ in {path}: "
            f"{fit_target.shape} vs {downstream_target.shape}"
        )
    return payload, fit_target, fit_predicted, downstream_target


def _one_split(
    fit_target: np.ndarray,
    fit_predicted: np.ndarray,
    downstream_target: np.ndarray,
    config: Config,
    split_seed: int,
) -> dict[str, Any]:
    train_idx, validation_idx, test_idx = _split_indices(fit_target.shape[0], config, split_seed)
    mean, basis = _fit_basis(fit_target[train_idx], config.rank)
    x_train = _project(fit_target[train_idx], mean, basis)
    y_train = _project(fit_predicted[train_idx], mean, basis)
    x_validation = _project(fit_target[validation_idx], mean, basis)
    y_validation = _project(fit_predicted[validation_idx], mean, basis)
    x_test = _project(fit_target[test_idx], mean, basis)
    y_test = _project(fit_predicted[test_idx], mean, basis)
    downstream_test = _project(downstream_target[test_idx], mean, basis)

    anchor = _fit_ridge(x_train, y_train, config.ridge)
    candidates = _candidate_family(anchor, config.candidate_count, config.seed + 100_000 * split_seed)
    covariance = x_train.T @ x_train / max(x_train.shape[0] - 1, 1)
    train_geometry = [_geometry(covariance, candidate, config.downstream_horizon) for candidate in candidates]
    train_alignment = np.asarray([item["alignment"] for item in train_geometry])
    train_schur = np.asarray([item["schur_offdiagonal_energy"] for item in train_geometry])
    train_departure = np.asarray([item["departure_from_normality"] for item in train_geometry])
    train_transient = np.asarray([item["transient_amplification"] for item in train_geometry])
    validation_one_step = np.asarray([_mse(x_validation @ candidate, y_validation) for candidate in candidates])
    test_metrics = [
        _evaluate(x_test, y_test, downstream_test, candidate, config.downstream_horizon)
        for candidate in candidates
    ]
    test_one_step = np.asarray([item["one_step_mse"] for item in test_metrics])
    test_transfer = np.asarray([item["downstream_transfer_mse"] for item in test_metrics])

    alignment_selected = int(np.argmin(train_alignment))
    validation_selected = int(np.argmin(validation_one_step))
    spectrum_rng = np.random.default_rng(config.seed + 200_000 * split_seed)
    spectrum_selected = int(spectrum_rng.integers(0, len(candidates)))
    oracle_selected = int(np.argmin(test_transfer))
    validation_floor = float(validation_one_step.min())
    feasible = np.flatnonzero(
        validation_one_step <= validation_floor * (1.0 + float(config.prediction_tolerance))
    )
    constrained_alignment_selected = int(feasible[np.argmin(train_alignment[feasible])])
    constrained_spectrum_selected = int(spectrum_rng.choice(feasible))
    geometry_rank = np.mean(np.stack([
        np.argsort(np.argsort(train_alignment)),
        np.argsort(np.argsort(train_schur)),
        np.argsort(np.argsort(train_departure)),
        np.argsort(np.argsort(train_transient)),
    ]), axis=0)
    geometry_selected = int(np.argmin(geometry_rank))
    constrained_geometry_selected = int(feasible[np.argmin(geometry_rank[feasible])])

    pair_left, pair_right = 0, 1
    best_validation_gap = float("inf")
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            gap = abs(float(validation_one_step[left] - validation_one_step[right]))
            if gap < best_validation_gap:
                best_validation_gap = gap
                pair_left, pair_right = left, right

    return {
        "split_seed": int(split_seed),
        "n_train": int(train_idx.size),
        "n_validation": int(validation_idx.size),
        "n_test": int(test_idx.size),
        "rank": int(basis.shape[0]),
        "candidate_count": len(candidates),
        "eigenvalue_match_max_abs": _eigenvalue_match(anchor, candidates),
        "alignment_selected": alignment_selected,
        "validation_selected": validation_selected,
        "spectrum_only_selected": spectrum_selected,
        "oracle_selected": oracle_selected,
        "alignment_selected_is_oracle": alignment_selected == oracle_selected,
        "spectrum_only_selected_is_oracle": spectrum_selected == oracle_selected,
        "alignment_selected_test": test_metrics[alignment_selected],
        "validation_selected_is_oracle": validation_selected == oracle_selected,
        "validation_selected_test": test_metrics[validation_selected],
        "spectrum_only_selected_test": test_metrics[spectrum_selected],
        "error_constrained_candidate_count": int(feasible.size),
        "error_constrained_alignment_selected": constrained_alignment_selected,
        "error_constrained_spectrum_only_selected": constrained_spectrum_selected,
        "geometry_selected": geometry_selected,
        "error_constrained_geometry_selected": constrained_geometry_selected,
        "error_constrained_alignment_selected_is_oracle": constrained_alignment_selected == oracle_selected,
        "error_constrained_spectrum_only_selected_is_oracle": constrained_spectrum_selected == oracle_selected,
        "geometry_selected_is_oracle": geometry_selected == oracle_selected,
        "error_constrained_geometry_selected_is_oracle": constrained_geometry_selected == oracle_selected,
        "error_constrained_alignment_selected_test": test_metrics[constrained_alignment_selected],
        "error_constrained_spectrum_only_selected_test": test_metrics[constrained_spectrum_selected],
        "oracle_test": test_metrics[oracle_selected],
        "alignment_selection_regret": float(test_transfer[alignment_selected] - test_transfer[oracle_selected]),
        "validation_selection_regret": float(test_transfer[validation_selected] - test_transfer[oracle_selected]),
        "spectrum_only_selection_regret": float(test_transfer[spectrum_selected] - test_transfer[oracle_selected]),
        "error_constrained_alignment_selection_regret": float(
            test_transfer[constrained_alignment_selected] - test_transfer[oracle_selected]
        ),
        "error_constrained_spectrum_only_selection_regret": float(
            test_transfer[constrained_spectrum_selected] - test_transfer[oracle_selected]
        ),
        "geometry_selection_regret": float(test_transfer[geometry_selected] - test_transfer[oracle_selected]),
        "error_constrained_geometry_selection_regret": float(
            test_transfer[constrained_geometry_selected] - test_transfer[oracle_selected]
        ),
        "alignment_transfer_rank_spearman": _rank_correlation(train_alignment, test_transfer),
        "matched_pair": {
            "left": int(pair_left),
            "right": int(pair_right),
            "validation_one_step_abs_gap": float(abs(validation_one_step[pair_left] - validation_one_step[pair_right])),
            "test_one_step_difference_left_minus_right": float(test_one_step[pair_left] - test_one_step[pair_right]),
            "test_downstream_difference_left_minus_right": float(test_transfer[pair_left] - test_transfer[pair_right]),
            "alignment_difference_left_minus_right": float(train_alignment[pair_left] - train_alignment[pair_right]),
        },
        "candidate_diagnostics": [
            {
                "candidate": int(index),
                "train_alignment": float(train_alignment[index]),
                "train_schur_offdiagonal_energy": float(train_schur[index]),
                "train_departure_from_normality": float(train_departure[index]),
                "train_transient_amplification": float(train_transient[index]),
                "validation_one_step_mse": float(validation_one_step[index]),
                "test_one_step_mse": float(test_one_step[index]),
                "test_downstream_transfer_mse": float(test_transfer[index]),
            }
            for index in range(len(candidates))
        ],
    }


def _mean_ci(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(20261008 + array.size)
    draws = array[rng.integers(0, array.size, size=(4000, array.size))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n": int(array.size),
    }


def run(inputs: dict[str, Path], config: Config) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    source_protocols: dict[str, Any] = {}
    for label, path in inputs.items():
        payload, fit_target, fit_predicted, downstream_target = _load_vectors(
            path, config.fit_horizon, config.downstream_horizon
        )
        source_protocols[label] = {
            "path": str(path),
            "checkpoint_protocol": payload.get("protocol", {}),
            "n_rows": int(fit_target.shape[0]),
            "latent_dimension": int(fit_target.shape[1]),
        }
        for split_seed in config.split_seeds:
            record = _one_split(fit_target, fit_predicted, downstream_target, config, int(split_seed))
            record["label"] = label
            records.append(record)

    def collect(name: str) -> list[float]:
        return [float(record[name]) for record in records]

    summary = {
        "alignment_selection_win_rate": _mean_ci([float(record["alignment_selected_is_oracle"]) for record in records]),
        "spectrum_only_selection_win_rate": _mean_ci([float(record["spectrum_only_selected_is_oracle"]) for record in records]),
        "error_constrained_alignment_selection_win_rate": _mean_ci([
            float(record["error_constrained_alignment_selected_is_oracle"]) for record in records
        ]),
        "error_constrained_spectrum_only_selection_win_rate": _mean_ci([
            float(record["error_constrained_spectrum_only_selected_is_oracle"]) for record in records
        ]),
        "alignment_selection_regret": _mean_ci(collect("alignment_selection_regret")),
        "spectrum_only_selection_regret": _mean_ci(collect("spectrum_only_selection_regret")),
        "error_constrained_alignment_selection_regret": _mean_ci(
            collect("error_constrained_alignment_selection_regret")
        ),
        "validation_selection_regret": _mean_ci(collect("validation_selection_regret")),
        "alignment_minus_validation_regret": _mean_ci([
            record["alignment_selection_regret"] - record["validation_selection_regret"]
            for record in records
        ]),
        "error_constrained_spectrum_only_selection_regret": _mean_ci(
            collect("error_constrained_spectrum_only_selection_regret")
        ),
        "geometry_selection_regret": _mean_ci(collect("geometry_selection_regret")),
        "error_constrained_geometry_selection_regret": _mean_ci(
            collect("error_constrained_geometry_selection_regret")
        ),
        "geometry_selection_win_rate": _mean_ci([
            float(record["geometry_selected_is_oracle"]) for record in records
        ]),
        "error_constrained_geometry_selection_win_rate": _mean_ci([
            float(record["error_constrained_geometry_selected_is_oracle"]) for record in records
        ]),
        "error_constrained_alignment_minus_spectrum_regret": _mean_ci([
            record["error_constrained_alignment_selection_regret"]
            - record["error_constrained_spectrum_only_selection_regret"]
            for record in records
        ]),
        "error_constrained_candidate_count": _mean_ci(collect("error_constrained_candidate_count")),
        "alignment_transfer_rank_spearman": _mean_ci(collect("alignment_transfer_rank_spearman")),
        "matched_pair_test_one_step_difference": _mean_ci([
            record["matched_pair"]["test_one_step_difference_left_minus_right"] for record in records
        ]),
        "matched_pair_test_downstream_difference": _mean_ci([
            record["matched_pair"]["test_downstream_difference_left_minus_right"] for record in records
        ]),
        "matched_pair_validation_one_step_gap": _mean_ci([
            record["matched_pair"]["validation_one_step_abs_gap"] for record in records
        ]),
        "eigenvalue_match_max_abs": _mean_ci(collect("eigenvalue_match_max_abs")),
    }
    source_summary: dict[str, Any] = {}
    for label in sorted({str(record["label"]) for record in records}):
        source_records = [record for record in records if record["label"] == label]
        source_summary[label] = {
            "n_splits": len(source_records),
            "alignment_selection_regret": _mean_ci(
                [record["alignment_selection_regret"] for record in source_records]
            ),
            "validation_selection_regret": _mean_ci(
                [record["validation_selection_regret"] for record in source_records]
            ),
            "alignment_minus_validation_regret": _mean_ci([
                record["alignment_selection_regret"] - record["validation_selection_regret"]
                for record in source_records
            ]),
            "error_constrained_alignment_selection_regret": _mean_ci(
                [record["error_constrained_alignment_selection_regret"] for record in source_records]
            ),
            "error_constrained_spectrum_only_selection_regret": _mean_ci(
                [record["error_constrained_spectrum_only_selection_regret"] for record in source_records]
            ),
            "geometry_selection_regret": _mean_ci(
                [record["geometry_selection_regret"] for record in source_records]
            ),
            "error_constrained_geometry_selection_regret": _mean_ci(
                [record["error_constrained_geometry_selection_regret"] for record in source_records]
            ),
            "error_constrained_alignment_minus_spectrum_regret": _mean_ci([
                record["error_constrained_alignment_selection_regret"]
                - record["error_constrained_spectrum_only_selection_regret"]
                for record in source_records
            ]),
            "error_constrained_candidate_count": _mean_ci(
                [record["error_constrained_candidate_count"] for record in source_records]
            ),
        }
    report = {
        "protocol": {
            "estimand": "alignment-aware selection among fixed-spectrum operators derived from a frozen learned checkpoint",
            "fit_horizon": config.fit_horizon,
            "downstream_horizon": config.downstream_horizon,
            "operator_fit": "train-split ridge map from target latent to predicted latent after train-PCA",
            "alignment_split": "train only",
            "pair_matching_split": "validation only",
            "transfer_evaluation_split": "test only",
            "candidate_family": "anchor operator and orthogonal conjugates with identical eigenvalue multiset",
            "candidate_count": config.candidate_count,
            "rank": config.rank,
            "ridge": config.ridge,
            "split_seeds": list(config.split_seeds),
            "model_selection_baseline": "random tie-break among equal-spectrum candidates",
            "prediction_error_tolerance": config.prediction_tolerance,
            "error_constrained_selection": "validation one-step MSE within the declared tolerance of the best candidate, then train-only alignment tie-break",
            "additional_diagnostics": "Schur off-diagonal energy, departure from normality, and finite-horizon transient amplification computed on train covariance",
            "uncertainty_unit": "bootstrap over split seeds; not independent episode-level replication",
        },
        "source_protocols": source_protocols,
        "source_summary": source_summary,
        "summary": summary,
        "records": records,
        "claim_boundary": [
            "This is evidence on frozen learned-checkpoint latent vectors and checkpoint-derived linear operator probes.",
            "The conjugated candidates are not independently trained neural checkpoints.",
            "The benchmark does not establish nonlinear LeWM training gains, physical causality, online planning, or universal world-model transfer.",
        ],
    }
    _finite(report)
    return report


def markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Learned-checkpoint covariance--predictor diagnostic benchmark",
        "",
        "The benchmark uses frozen JEPA-WM/DINO-WM checkpoint vectors and a train/validation/test protocol.",
        "Candidate operators are an anchor ridge fit and spectrum-preserving orthogonal conjugates; they are diagnostic probes, not retrained neural checkpoints.",
        "",
        "## Primary results",
        "",
        "| Quantity | Mean | 95% bootstrap CI |",
        "| --- | ---: | ---: |",
    ]
    names = [
        ("Alignment selection win rate", "alignment_selection_win_rate"),
        ("Spectrum-only selection win rate", "spectrum_only_selection_win_rate"),
        ("Error-constrained alignment win rate", "error_constrained_alignment_selection_win_rate"),
        ("Error-constrained spectrum-only win rate", "error_constrained_spectrum_only_selection_win_rate"),
        ("Alignment selection regret", "alignment_selection_regret"),
        ("Validation one-step-error selection regret", "validation_selection_regret"),
        ("Alignment minus validation regret", "alignment_minus_validation_regret"),
        ("Spectrum-only selection regret", "spectrum_only_selection_regret"),
        ("Error-constrained alignment regret", "error_constrained_alignment_selection_regret"),
        ("Error-constrained spectrum-only regret", "error_constrained_spectrum_only_selection_regret"),
        ("Geometry composite regret", "geometry_selection_regret"),
        ("Error-constrained geometry composite regret", "error_constrained_geometry_selection_regret"),
        ("Error-constrained alignment minus spectrum regret", "error_constrained_alignment_minus_spectrum_regret"),
        ("Error-constrained candidate count", "error_constrained_candidate_count"),
        ("Alignment--test-transfer Spearman", "alignment_transfer_rank_spearman"),
        ("Matched pair test one-step difference", "matched_pair_test_one_step_difference"),
        ("Matched pair test downstream difference", "matched_pair_test_downstream_difference"),
        ("Matched pair validation one-step gap", "matched_pair_validation_one_step_gap"),
        ("Maximum eigenvalue mismatch", "eigenvalue_match_max_abs"),
    ]
    for label, key in names:
        value = summary[key]
        lines.append(f"| {label} | {value['mean']:.6g} | [{value['ci_low']:.6g}, {value['ci_high']:.6g}] |")
    lines.extend([
        "",
        "## Stratified results",
        "",
        "The pooled comparison is decomposed by checkpoint and environment below. "
        "Intervals are bootstrap intervals over split seeds within each source; these are "
        "stability intervals, not independent episode-level confidence intervals. "
        "The candidate count is the number retained by the validation-error constraint.",
        "",
        "| Checkpoint / environment | Alignment regret | Validation-error regret | Alignment minus validation regret | Constrained alignment regret | Constrained spectrum-only regret | Alignment minus spectrum regret | Feasible candidates |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for label, source in report["source_summary"].items():
        def fmt(key: str) -> str:
            value = source[key]
            return f"{value['mean']:.4g} [{value['ci_low']:.4g}, {value['ci_high']:.4g}]"

        lines.append(
            f"| {label} | {fmt('alignment_selection_regret')} | "
            f"{fmt('validation_selection_regret')} | "
            f"{fmt('alignment_minus_validation_regret')} | "
            f"{fmt('error_constrained_alignment_selection_regret')} | "
            f"{fmt('error_constrained_spectrum_only_selection_regret')} | "
            f"{fmt('error_constrained_alignment_minus_spectrum_regret')} | "
            f"{fmt('error_constrained_candidate_count')} |"
        )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]], ""])
    return "\n".join(lines)


def _parse_inputs(values: list[str]) -> dict[str, Path]:
    inputs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--input must be LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        inputs[label] = Path(raw_path)
    if not inputs:
        raise ValueError("at least one --input is required")
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="LABEL=metrics.json; repeatable")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--fit-horizon", type=int, default=1)
    parser.add_argument("--downstream-horizon", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--candidate-count", type=int, default=17)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    config = Config(
        fit_horizon=args.fit_horizon,
        downstream_horizon=args.downstream_horizon,
        rank=args.rank,
        ridge=args.ridge,
        candidate_count=args.candidate_count,
        prediction_tolerance=0.05,
        split_seeds=tuple(args.split_seeds),
    )
    inputs = _parse_inputs(args.input)
    report = run(inputs, config)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "learned_checkpoint_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "learned_checkpoint_diagnostic.md").write_text(markdown(report))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
