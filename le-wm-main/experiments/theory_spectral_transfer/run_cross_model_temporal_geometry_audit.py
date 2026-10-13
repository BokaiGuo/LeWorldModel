"""Cross-model held-out temporal operator-geometry audit.

This consumes already generated frozen-checkpoint temporal metrics containing
per-episode target/predicted latent vectors.  For every model/environment and
horizon, a ridge operator is fitted on an internal split and evaluated on the
remaining episodes.  The audit compares covariance--operator commutator,
covariance/operator subspace overlap, non-normality, and held-out residual.

The result is a cross-model diagnostic, not a universal model ranking: the
inputs are frozen checkpoint artifacts and the supplied temporal protocols are
not online control evaluations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUTS = {
    "JEPA-WM/PushT": "results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate/jepa_wm_temporal_metrics.json",
    "DINO-WM/PushT": "results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate/dino_wm_pusht_temporal_metrics.json",
    "JEPA-WM/PointMaze": "results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate/jepa_wm_pointmaze_temporal_metrics_200.json",
    "DINO-WM/PointMaze": "results/theory_operator_alignment_iclr_oral_20260829/jepa_wm_temporal_gate/dino_wm_pointmaze_temporal_metrics_200.json",
}


def _finite(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"non-finite value: {value}")


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return {"n": int(x.size), "pearson": 0.0, "spearman": 0.0}
    return {
        "n": int(x.size),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(_rank(x), _rank(y))[0, 1]),
    }


def _fit_operator(
    target_train: np.ndarray,
    predicted_train: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_mean = target_train.mean(axis=0)
    predicted_mean = predicted_train.mean(axis=0)
    centered_target = target_train - target_mean
    centered_predicted = predicted_train - predicted_mean
    dimension = target_train.shape[1]
    gram = (centered_target.T @ centered_target) / max(target_train.shape[0], 1)
    scale = max(float(np.trace(gram)) / max(dimension, 1), 1e-8)
    regularizer = float(ridge) * scale * np.eye(dimension, dtype=np.float64)
    operator = np.linalg.solve(
        gram + regularizer,
        (centered_target.T @ centered_predicted) / max(target_train.shape[0], 1),
    )
    return operator, target_mean, predicted_mean


def _geometry(
    covariance: np.ndarray,
    operator: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    commutator = covariance @ operator - operator @ covariance
    covariance_norm = float(np.linalg.norm(covariance, ord="fro"))
    operator_norm = float(np.linalg.norm(operator, ord="fro"))
    normalized_commutator = float(
        np.linalg.norm(commutator, ord="fro")
        / max(covariance_norm * operator_norm, 1e-12)
    )
    covariance_eigenvalues, covariance_vectors = np.linalg.eigh(covariance)
    covariance_eigenvalues = np.maximum(covariance_eigenvalues, 0.0)
    covariance_order = np.argsort(covariance_eigenvalues)[::-1]
    covariance_eigenvalues = covariance_eigenvalues[covariance_order]
    covariance_vectors = covariance_vectors[:, covariance_order]
    _, singular_values, right_vectors_t = np.linalg.svd(operator, full_matrices=False)
    k = min(int(top_k), covariance_vectors.shape[1], right_vectors_t.shape[0])
    covariance_subspace = covariance_vectors[:, :k]
    operator_subspace = right_vectors_t.T[:, :k]
    subspace_singular_values = np.linalg.svd(
        covariance_subspace.T @ operator_subspace, compute_uv=False
    )
    subspace_singular_values = np.clip(subspace_singular_values, 0.0, 1.0)
    stable_rank = float(
        np.square(covariance_eigenvalues.sum())
        / max(np.square(covariance_eigenvalues).sum(), 1e-12)
    )
    operator_radius = float(np.max(np.abs(np.linalg.eigvals(operator))))
    operator_spectral_norm = float(singular_values[0]) if singular_values.size else 0.0
    return {
        "normalized_commutator": normalized_commutator,
        "top_subspace_overlap": float(np.square(subspace_singular_values).mean()),
        "max_principal_angle_degrees": float(
            np.degrees(np.arccos(subspace_singular_values.min()))
        ),
        "covariance_stable_rank": stable_rank,
        "operator_non_normality": float(
            np.linalg.norm(operator @ operator.T - operator.T @ operator, ord="fro")
            / max(operator_norm * operator_norm, 1e-12)
        ),
        "operator_spectral_radius": operator_radius,
        "operator_spectral_norm": operator_spectral_norm,
        "singular_to_radius": operator_spectral_norm / max(operator_radius, 1e-12),
    }


def _load_vectors(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    payload = json.loads(path.read_text())
    vectors = payload.get("operator_fit_vectors")
    if not isinstance(vectors, dict) or "target" not in vectors or "predicted" not in vectors:
        raise ValueError(f"{path} has no operator_fit_vectors target/predicted arrays")
    targets = {
        str(horizon): np.asarray(values, dtype=np.float64)
        for horizon, values in vectors["target"].items()
    }
    predicted = {
        str(horizon): np.asarray(values, dtype=np.float64)
        for horizon, values in vectors["predicted"].items()
    }
    for horizon in sorted(targets):
        if horizon not in predicted:
            raise ValueError(f"missing predicted vectors for horizon {horizon} in {path}")
        if targets[horizon].shape != predicted[horizon].shape or targets[horizon].ndim != 2:
            raise ValueError(f"shape mismatch at horizon {horizon} in {path}")
        if targets[horizon].shape[0] < 6:
            raise ValueError(f"too few episodes at horizon {horizon} in {path}")
    return payload, targets, predicted


def _one_record(
    label: str,
    source_path: Path,
    horizon: str,
    target: np.ndarray,
    predicted: np.ndarray,
    split_seed: int,
    train_fraction: float,
    ridge: float,
    top_k: int,
) -> dict[str, Any]:
    n = target.shape[0]
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(n)
    train_count = min(n - 2, max(3, int(round(n * train_fraction))))
    train_indices = np.sort(order[:train_count])
    holdout_indices = np.sort(order[train_count:])
    operator, target_mean, predicted_mean = _fit_operator(
        target[train_indices], predicted[train_indices], ridge
    )
    target_train = target[train_indices] - target_mean
    covariance = (target_train.T @ target_train) / max(target_train.shape[0] - 1, 1)
    holdout_prediction = (
        (target[holdout_indices] - target_mean) @ operator + predicted_mean
    )
    holdout_residual_mse = float(
        np.mean(np.square(holdout_prediction - predicted[holdout_indices]))
    )
    result = {
        "label": label,
        "source": str(source_path),
        "horizon": int(horizon),
        "split_seed": int(split_seed),
        "n_total": int(n),
        "n_train": int(train_indices.size),
        "n_holdout": int(holdout_indices.size),
        "holdout_residual_mse": holdout_residual_mse,
        "geometry": _geometry(covariance, operator, top_k),
    }
    _finite(result)
    return result


def run_audit(
    inputs: dict[str, Path],
    split_seeds: list[int],
    train_fraction: float = 2.0 / 3.0,
    ridge: float = 1e-3,
    top_k: int = 8,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    source_protocols: dict[str, Any] = {}
    for label, path in inputs.items():
        payload, targets, predicted = _load_vectors(path)
        source_protocols[label] = {
            "path": str(path),
            "protocol": payload.get("protocol", {}),
            "horizons": sorted(targets),
            "episodes_by_horizon": {horizon: int(values.shape[0]) for horizon, values in targets.items()},
            "latent_dimension": {horizon: int(values.shape[1]) for horizon, values in targets.items()},
        }
        for horizon in sorted(targets, key=int):
            for split_seed in split_seeds:
                records.append(
                    _one_record(
                        label,
                        path,
                        horizon,
                        targets[horizon],
                        predicted[horizon],
                        int(split_seed),
                        train_fraction,
                        ridge,
                        top_k,
                    )
                )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(f"{record['label']}::h{record['horizon']}", []).append(record)
    summary: dict[str, Any] = {}
    for key, rows in grouped.items():
        summary[key] = {
            "label": rows[0]["label"],
            "horizon": rows[0]["horizon"],
            "n_splits": len(rows),
            "metrics": {
                metric: {
                    "mean": float(np.mean([row[metric] if metric == "holdout_residual_mse" else row["geometry"][metric] for row in rows])),
                    "std": float(np.std([row[metric] if metric == "holdout_residual_mse" else row["geometry"][metric] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
                }
                for metric in (
                    "holdout_residual_mse",
                    "normalized_commutator",
                    "top_subspace_overlap",
                    "max_principal_angle_degrees",
                    "operator_non_normality",
                    "singular_to_radius",
                )
            },
        }
    commutators = np.asarray([row["geometry"]["normalized_commutator"] for row in records], dtype=np.float64)
    residuals = np.asarray([row["holdout_residual_mse"] for row in records], dtype=np.float64)
    report = {
        "protocol": {
            "kind": "frozen cross-model held-out temporal operator geometry",
            "split_seeds": [int(seed) for seed in split_seeds],
            "train_fraction": train_fraction,
            "ridge": ridge,
            "top_k": top_k,
            "operator_fit": "ridge linear map from target latent to predicted latent on split-train",
            "geometry_fit_split": "split-train covariance and operator",
            "residual_split": "split-holdout prediction residual",
            "source_models": sorted(inputs),
        },
        "source_protocols": source_protocols,
        "records": records,
        "summary": summary,
        "cross_model_association": {
            "commutator_vs_holdout_residual": _correlation(commutators, residuals),
        },
        "claim_boundary": [
            "this is a frozen-checkpoint temporal diagnostic using supplied predicted/target vectors",
            "geometry is fit on internal split-train data and residual is evaluated on split-holdout data",
            "cross-model association is descriptive and does not establish a universal model-quality scalar",
            "the audit does not provide online planning, causal-control, or end-to-end LeWM training evidence",
        ],
    }
    _finite(report)
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Cross-model held-out temporal operator geometry",
        "",
        "Operators are fitted on internal split-train target/predicted latent vectors; residuals are evaluated on split-holdout vectors. Values are mean +/- standard deviation over split seeds.",
        "",
        "| model/environment | h | splits | holdout residual MSE | normalized commutator | top-8 subspace overlap | max principal angle | non-normality | singular/radius |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in sorted(report["summary"].items()):
        metrics = values["metrics"]
        def fmt(name: str) -> str:
            return f"{metrics[name]['mean']:.6g} +/- {metrics[name]['std']:.2g}"
        lines.append(
            f"| {values['label']} | {values['horizon']} | {values['n_splits']} | "
            f"{fmt('holdout_residual_mse')} | {fmt('normalized_commutator')} | "
            f"{fmt('top_subspace_overlap')} | {fmt('max_principal_angle_degrees')} | "
            f"{fmt('operator_non_normality')} | {fmt('singular_to_radius')} |"
        )
    association = report["cross_model_association"]["commutator_vs_holdout_residual"]
    lines.extend(
        [
            "",
            f"Across {association['n']} model/environment-horizon-split cells, Pearson/Spearman correlation between normalized commutator and held-out residual MSE is {association['pearson']:.4f}/{association['spearman']:.4f}.",
            "",
            "## Boundary",
            "",
            *[f"- {item}" for item in report["claim_boundary"]],
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def write_figure(report: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "JEPA-WM/PushT": "#1b9e77",
        "DINO-WM/PushT": "#d95f02",
        "JEPA-WM/PointMaze": "#7570b3",
        "DINO-WM/PointMaze": "#e7298a",
    }
    fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for label in sorted({str(row["label"]) for row in report["records"]}):
        rows = [row for row in report["records"] if row["label"] == label]
        axis.scatter(
            [row["geometry"]["normalized_commutator"] for row in rows],
            [row["holdout_residual_mse"] for row in rows],
            label=label,
            color=colors.get(label),
            alpha=0.78,
            s=38,
        )
    association = report["cross_model_association"]["commutator_vs_holdout_residual"]
    axis.set_xlabel("normalized covariance--operator commutator")
    axis.set_ylabel("held-out residual MSE")
    axis.set_title(
        "Cross-model temporal geometry: pooled Pearson "
        f"{association['pearson']:.3f}"
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _parse_inputs(values: list[str] | None, repo_root: Path) -> dict[str, Path]:
    if not values:
        return {label: repo_root / path for label, path in DEFAULT_INPUTS.items()}
    inputs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--input must be LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_root / path
        inputs[label] = path
    return inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--input", action="append", help="LABEL=JSON_PATH; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    inputs = _parse_inputs(args.input, args.repo_root)
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing input artifacts: {missing}")
    report = run_audit(
        inputs,
        args.split_seeds,
        train_fraction=args.train_fraction,
        ridge=args.ridge,
        top_k=args.top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    write_figure(report, args.output.with_suffix(".png"))
    (args.output.parent / "README.md").write_text(
        "# Cross-model held-out temporal operator geometry\n\n"
        "This artifact compares frozen JEPA-WM and DINO-WM temporal vectors using internal held-out operator fits.\n"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
