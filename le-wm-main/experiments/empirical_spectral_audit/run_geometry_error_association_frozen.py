"""Associate local operator geometry with held-out frozen prediction error.

For each action-norm bin, a linear operator is fitted on one half of frozen
PushT latent rows and evaluated on the other half.  The audit records
commutator, non-normality, singular-value and h=8 transient-gain diagnostics
alongside held-out residual MSE.  Associations are descriptive cross-fit
evidence, not causal claims about rollout or control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from experiments.empirical_spectral_audit.run_nonnormality_control_frozen import (
        _operator_diagnostics,
    )
    from experiments.empirical_spectral_audit.run_whitening_invariance_frozen import (
        _finite,
        _fit_operator,
    )
except ModuleNotFoundError:  # direct invocation from the script directory
    from run_nonnormality_control_frozen import _operator_diagnostics  # type: ignore
    from run_whitening_invariance_frozen import _finite, _fit_operator  # type: ignore


def _labels(actions: np.ndarray, bins: int) -> np.ndarray:
    score = np.linalg.norm(actions, axis=1)
    thresholds = np.quantile(score, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    return np.digitize(score, thresholds, right=False).astype(np.int64)


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 3 or right.size != left.size:
        return None
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return _corr(_rank(left), _rank(right))


def _partial_corr(
    left: np.ndarray,
    right: np.ndarray,
    control: np.ndarray,
    rank_transform: bool = False,
) -> float | None:
    if left.size < 4 or right.size != left.size or control.size != left.size:
        return None
    if rank_transform:
        left, right, control = _rank(left), _rank(right), _rank(control)
    design = np.column_stack([np.ones(left.size), control])
    left_residual = left - design @ np.linalg.lstsq(design, left, rcond=None)[0]
    right_residual = right - design @ np.linalg.lstsq(design, right, rcond=None)[0]
    return _corr(left_residual, right_residual)


def _one_record(
    source: np.ndarray,
    target: np.ndarray,
    actions: np.ndarray,
    seed: int,
    bins: int,
    ridge: float,
) -> dict[str, Any]:
    labels = _labels(actions, bins)
    groups = []
    for label in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == label)
        if indices.size < 8:
            continue
        order = np.random.default_rng(seed + 17 * (int(label) + 1)).permutation(indices)
        cut = indices.size // 2
        train, test = order[:cut], order[cut:]
        covariance, operator = _fit_operator(source[train], target[train], ridge)
        source_mean = source[train].mean(axis=0)
        target_mean = target[train].mean(axis=0)
        prediction = (source[test] - source_mean) @ operator.T + target_mean
        diagnostics = _operator_diagnostics(covariance, operator, [8])
        groups.append(
            {
                "label": int(label),
                "train_n": int(train.size),
                "test_n": int(test.size),
                "heldout_residual_mse": float(np.square(prediction - target[test]).mean()),
                "relative_heldout_residual_mse": float(
                    np.square(prediction - target[test]).mean()
                    / max(float(np.square(target[test]).mean()), 1e-12)
                ),
                "mean_action_norm": float(np.linalg.norm(actions[test], axis=1).mean()),
                "commutator": float(diagnostics["commutator"]["normalized_frobenius"]),
                "nonnormality": float(diagnostics["henrici_nonnormality"]),
                "singular_to_radius": float(diagnostics["singular_to_radius_ratio"]),
                "gain_over_radius_h8": float(
                    diagnostics["operator_powers"]["8"]["gain_over_radius_power"]
                ),
            }
        )
    mse = np.asarray([group["heldout_residual_mse"] for group in groups], dtype=np.float64)
    relative_mse = np.asarray(
        [group["relative_heldout_residual_mse"] for group in groups], dtype=np.float64
    )
    action_norm = np.asarray([group["mean_action_norm"] for group in groups], dtype=np.float64)
    metrics = {
        "commutator": np.asarray([group["commutator"] for group in groups], dtype=np.float64),
        "nonnormality": np.asarray([group["nonnormality"] for group in groups], dtype=np.float64),
        "singular_to_radius": np.asarray(
            [group["singular_to_radius"] for group in groups], dtype=np.float64
        ),
        "gain_over_radius_h8": np.asarray(
            [group["gain_over_radius_h8"] for group in groups], dtype=np.float64
        ),
    }
    correlations = {}
    for name, values in metrics.items():
        correlations[name] = {
            "pearson": _corr(values, mse),
            "spearman": _spearman(values, mse),
            "relative_mse_pearson": _corr(values, relative_mse),
            "relative_mse_spearman": _spearman(values, relative_mse),
            "partial_pearson_action_norm": _partial_corr(values, mse, action_norm),
            "partial_spearman_action_norm": _partial_corr(
                values, mse, action_norm, rank_transform=True
            ),
        }
    return {
        "seed": int(seed),
        "num_samples": int(source.shape[0]),
        "bins": int(bins),
        "num_groups": len(groups),
        "group_results": groups,
        "correlations_with_heldout_mse": correlations,
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    names = ["commutator", "nonnormality", "singular_to_radius", "gain_over_radius_h8"]
    output = {}
    for name in names:
        pearson = np.asarray(
            [record["correlations_with_heldout_mse"][name]["pearson"] for record in records if record["correlations_with_heldout_mse"][name]["pearson"] is not None],
            dtype=np.float64,
        )
        spearman = np.asarray(
            [record["correlations_with_heldout_mse"][name]["spearman"] for record in records if record["correlations_with_heldout_mse"][name]["spearman"] is not None],
            dtype=np.float64,
        )
        partial_pearson = np.asarray(
            [record["correlations_with_heldout_mse"][name]["partial_pearson_action_norm"] for record in records if record["correlations_with_heldout_mse"][name]["partial_pearson_action_norm"] is not None],
            dtype=np.float64,
        )
        partial_spearman = np.asarray(
            [record["correlations_with_heldout_mse"][name]["partial_spearman_action_norm"] for record in records if record["correlations_with_heldout_mse"][name]["partial_spearman_action_norm"] is not None],
            dtype=np.float64,
        )
        output[name] = {
            "num_configurations": int(pearson.size),
            "pearson_mean": float(pearson.mean()) if pearson.size else None,
            "pearson_std": float(pearson.std(ddof=1)) if pearson.size > 1 else None,
            "spearman_mean": float(spearman.mean()) if spearman.size else None,
            "spearman_std": float(spearman.std(ddof=1)) if spearman.size > 1 else None,
            "partial_pearson_mean": float(partial_pearson.mean()) if partial_pearson.size else None,
            "partial_pearson_std": float(partial_pearson.std(ddof=1)) if partial_pearson.size > 1 else None,
            "partial_spearman_mean": float(partial_spearman.mean()) if partial_spearman.size else None,
            "partial_spearman_std": float(partial_spearman.std(ddof=1)) if partial_spearman.size > 1 else None,
            "pearson_positive_configurations": int(np.sum(pearson > 0.0)) if pearson.size else 0,
        }
    return output


def run_report(
    source: Path,
    output: Path,
    sample_caps: list[int],
    seeds: list[int],
    bins: list[int],
    ridge: float,
) -> dict[str, Any]:
    with np.load(source, allow_pickle=False) as archive:
        required = {"latents", "predicted_dynamics", "actions"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"missing arrays: {sorted(missing)}")
        latents = np.asarray(archive["latents"], dtype=np.float64)
        predicted = np.asarray(archive["predicted_dynamics"], dtype=np.float64)
        actions = np.asarray(archive["actions"], dtype=np.float64)
    if latents.shape != predicted.shape or latents.shape[0] != actions.shape[0]:
        raise ValueError("source arrays have incompatible shapes")
    target = latents + predicted
    records = []
    for seed in seeds:
        for cap in sample_caps:
            if cap <= 1 or cap > latents.shape[0]:
                raise ValueError(f"invalid sample cap {cap}")
            rows = np.sort(np.random.default_rng(seed).choice(latents.shape[0], size=cap, replace=False))
            for num_bins in bins:
                record = _one_record(latents[rows], target[rows], actions[rows], seed, num_bins, ridge)
                records.append(record)
                print(
                    f"seed={seed} rows={cap} bins={num_bins} "
                    f"comm_spearman={record['correlations_with_heldout_mse']['commutator']['spearman']} "
                    f"nonnormal_spearman={record['correlations_with_heldout_mse']['nonnormality']['spearman']}",
                    flush=True,
                )
    report = {
        "artifact": "frozen_geometry_error_association",
        "status": "frozen_offline_boundary_evidence",
        "source": str(source),
        "protocol": {
            "sample_caps": [int(value) for value in sample_caps],
            "seeds": [int(value) for value in seeds],
            "action_bins": [int(value) for value in bins],
            "fit": "within-bin centered ridge operator on one half; held-out residual on the other half",
            "metrics": ["commutator", "Henrici non-normality", "singular/radius ratio", "h=8 gain/radius^8"],
            "association": "descriptive raw and action-norm-controlled Pearson/Spearman correlations across action bins",
            "rollout_and_planning_available": False,
        },
        "records": records,
        "aggregate_association": _aggregate(records),
        "claim_boundary": [
        "These are cross-fit associations between local operator diagnostics and one-step held-out residual MSE.",
        "Partial correlations control the mean action norm of each held-out action bin; they are still descriptive and low-degree-of-freedom.",
            "They do not establish that any geometry metric causes rollout or planning performance.",
            "The source artifact lacks episode/timestep and multi-horizon ground-truth fields.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Local geometry versus held-out residual error on frozen LeWM latents",
        "",
        "Each action-bin operator is fitted on one half and evaluated on the other. Correlations are descriptive across bins within each configuration.",
        "",
        "## Across-configuration association",
        "",
        "| metric | configurations | Pearson mean +/- SD | Spearman mean +/- SD | partial Spearman mean +/- SD | positive Pearson configs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in report["aggregate_association"].items():
        pearson = (
            "n/a"
            if summary["pearson_mean"] is None
            else f"{summary['pearson_mean']:.4f} +/- {summary['pearson_std']:.4f}"
        )
        spearman = (
            "n/a"
            if summary["spearman_mean"] is None
            else f"{summary['spearman_mean']:.4f} +/- {summary['spearman_std']:.4f}"
        )
        partial_spearman = (
            "n/a"
            if summary["partial_spearman_mean"] is None
            else f"{summary['partial_spearman_mean']:.4f} +/- {summary['partial_spearman_std']:.4f}"
        )
        lines.append(
            f"| {name} | {summary['num_configurations']} | {pearson} | {spearman} | {partial_spearman} | "
            f"{summary['pearson_positive_configurations']} |"
        )
    lines.extend(
        [
            "",
            "## Per-configuration correlations",
            "",
            "| seed | rows | bins | commutator partial Spearman | non-normality partial Spearman | singular/radius partial Spearman | h=8 gain partial Spearman |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in report["records"]:
        corr = record["correlations_with_heldout_mse"]
        values = [
            corr[name]["partial_spearman_action_norm"]
            for name in ["commutator", "nonnormality", "singular_to_radius", "gain_over_radius_h8"]
        ]
        formatted = ["n/a" if value is None else f"{value:.4f}" for value in values]
        lines.append(
            f"| {record['seed']} | {record['num_samples']} | {record['bins']} | "
            + " | ".join(formatted)
            + " |"
        )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-caps", type=int, nargs="+", default=[100000, 200000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[3072, 3073, 3074])
    parser.add_argument("--bins", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()
    run_report(args.source, args.output, args.sample_caps, args.seeds, args.bins, args.ridge)
    print(args.output)
    print(args.output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
