"""Held-out readout probes for frozen LeWM covariance subspaces.

This audit asks what the covariance-relative latent bands actually encode. A
covariance basis is fitted on the training split only, latent features are
projected to full/top/middle/tail subspaces, and ridge probes are evaluated on
held-out rows for state, action, dynamics, predicted dynamics, and prediction
residual targets. Dimension-matched random subspaces provide a coordinate and
subspace null.

The source artifact is a frozen latent export. The result is a representation
readout diagnostic, not evidence of retraining, causal control, or planning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


TARGET_FIELDS = (
    "states",
    "actions",
    "dynamics",
    "predicted_dynamics",
    "prediction_residual",
)


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)):
        return bool(np.isfinite(value))
    return True


def _band_indices(dim: int, kind: str, k: int) -> np.ndarray:
    if k <= 0 or k > dim:
        raise ValueError(f"invalid k={k} for dim={dim}")
    if kind == "full":
        return np.arange(dim)
    if kind == "top":
        return np.arange(k)
    if kind == "middle":
        start = (dim - k) // 2
        return np.arange(start, start + k)
    if kind == "tail":
        return np.arange(dim - k, dim)
    raise ValueError(f"unsupported band kind: {kind}")


def _orthonormal_subspace(dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.standard_normal((dim, k))
    basis, _ = np.linalg.qr(matrix, mode="reduced")
    return basis


def _ridge_probe(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
    ridge: float,
) -> dict[str, float]:
    x_train = np.asarray(train_features, dtype=np.float64)
    x_test = np.asarray(test_features, dtype=np.float64)
    y_train = np.asarray(train_target, dtype=np.float64)
    y_test = np.asarray(test_target, dtype=np.float64)
    if x_train.ndim != 2 or x_test.ndim != 2 or y_train.ndim != 2 or y_test.ndim != 2:
        raise ValueError("probe inputs must be two-dimensional")
    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = y_train.mean(axis=0, keepdims=True)
    x_train = x_train - x_mean
    x_test = x_test - x_mean
    y_train = y_train - y_mean
    gram = x_train.T @ x_train
    alpha = float(ridge) * max(float(np.trace(gram)) / max(x_train.shape[1], 1), 1e-12)
    coefficients = np.linalg.solve(
        gram + alpha * np.eye(x_train.shape[1]), x_train.T @ y_train
    )
    prediction = x_test @ coefficients + y_mean
    residual = prediction - np.asarray(test_target, dtype=np.float64)
    sse = float(np.square(residual).sum())
    centered_test = np.asarray(test_target, dtype=np.float64) - np.asarray(
        train_target, dtype=np.float64
    ).mean(axis=0, keepdims=True)
    total = float(np.square(centered_test).sum())
    return {
        "r2": float(1.0 - sse / max(total, 1e-12)),
        "mse": float(np.mean(np.square(residual))),
        "alpha": alpha,
        "feature_dim": int(x_train.shape[1]),
    }


def _probe_subspace(
    z_train: np.ndarray,
    z_test: np.ndarray,
    target_train: np.ndarray,
    target_test: np.ndarray,
    basis: np.ndarray,
    indices: np.ndarray,
    ridge: float,
) -> dict[str, float]:
    train_features = (z_train @ basis[:, indices]).astype(np.float64, copy=False)
    test_features = (z_test @ basis[:, indices]).astype(np.float64, copy=False)
    return _ridge_probe(
        train_features, test_features, target_train, target_test, ridge=ridge
    )


def run_report(
    latents_path: Path,
    output: Path,
    max_samples: int,
    test_fraction: float,
    sample_seed: int,
    split_seed: int,
    ks: list[int],
    random_repeats: int,
    ridge: float,
) -> dict[str, Any]:
    with np.load(latents_path, allow_pickle=False) as archive:
        available = set(archive.files)
        required = {"latents", *TARGET_FIELDS}
        missing = sorted(required - available)
        if missing:
            raise KeyError(f"missing fields: {missing}")
        row_count = int(archive["latents"].shape[0])
        if max_samples <= 0 or max_samples >= row_count:
            selected = np.arange(row_count)
        else:
            rng = np.random.default_rng(sample_seed)
            selected = np.sort(rng.choice(row_count, size=max_samples, replace=False))
        latent = np.asarray(archive["latents"][selected], dtype=np.float64)
        targets = {
            name: np.asarray(archive[name][selected], dtype=np.float64)
            for name in TARGET_FIELDS
        }

    rng = np.random.default_rng(split_seed)
    permutation = rng.permutation(latent.shape[0])
    test_count = max(1, int(round(latent.shape[0] * float(test_fraction))))
    test_indices = permutation[:test_count]
    train_indices = permutation[test_count:]
    z_train = latent[train_indices]
    z_test = latent[test_indices]
    train_targets = {name: values[train_indices] for name, values in targets.items()}
    test_targets = {name: values[test_indices] for name, values in targets.items()}

    mean = z_train.mean(axis=0, keepdims=True)
    centered = z_train - mean
    covariance = centered.T @ centered / max(z_train.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    basis = eigenvectors[:, order]
    dim = int(latent.shape[1])
    ks = sorted({int(k) for k in ks})
    if any(k <= 0 or k > dim for k in ks):
        raise ValueError(f"ks must lie in [1, {dim}]")

    bands: list[tuple[str, int, np.ndarray]] = [("full", dim, np.arange(dim))]
    for k in ks:
        for kind in ("top", "middle", "tail"):
            bands.append((kind, k, _band_indices(dim, kind, k)))

    records: list[dict[str, Any]] = []
    random_null: dict[str, dict[str, dict[str, float]]] = {}
    for target_name in TARGET_FIELDS:
        for band_name, k, indices in bands:
            metrics = _probe_subspace(
                z_train,
                z_test,
                train_targets[target_name],
                test_targets[target_name],
                basis,
                indices,
                ridge,
            )
            records.append(
                {
                    "target": target_name,
                    "subspace": band_name,
                    "k": int(k),
                    **metrics,
                }
            )
        for k in ks:
            for target_kind in ("top", "middle", "tail"):
                values: list[float] = []
                for repeat in range(random_repeats):
                    random_basis = _orthonormal_subspace(
                        dim, k, np.random.default_rng(sample_seed + 100_000 * k + repeat)
                    )
                    metrics = _ridge_probe(
                        z_train @ random_basis,
                        z_test @ random_basis,
                        train_targets[target_name],
                        test_targets[target_name],
                        ridge,
                    )
                    values.append(metrics["r2"])
                random_null.setdefault(target_name, {})[f"{target_kind}{k}"] = {
                    "r2_mean": float(np.mean(values)),
                    "r2_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "r2_min": float(np.min(values)),
                    "r2_max": float(np.max(values)),
                    "repeats": int(random_repeats),
                }

    total_variance = float(eigenvalues.sum())
    stable_rank = float(total_variance**2 / max(float(np.square(eigenvalues).sum()), 1e-12))
    report = {
        "artifact": "frozen_lewm_covariance_subspace_probes",
        "status": "frozen_readout_diagnostic",
        "source": {
            "latents": str(latents_path),
            "source_rows": row_count,
            "selected_rows": int(latent.shape[0]),
            "latent_dim": dim,
            "fields": ["latents", *TARGET_FIELDS],
        },
        "protocol": {
            "sample_seed": int(sample_seed),
            "split_seed": int(split_seed),
            "test_fraction": float(test_fraction),
            "train_rows": int(z_train.shape[0]),
            "test_rows": int(z_test.shape[0]),
            "ks": ks,
            "ridge": float(ridge),
            "random_subspace_repeats": int(random_repeats),
            "basis": "training-split latent covariance eigenbasis",
            "readout": "centered ridge regression; R2 evaluated on held-out rows",
        },
        "covariance": {
            "stable_rank": stable_rank,
            "top_variance_share": {
                str(k): float(eigenvalues[:k].sum() / max(total_variance, 1e-12))
                for k in ks
            },
            "eigenvalues": eigenvalues.tolist(),
        },
        "records": records,
        "random_subspace_null": random_null,
        "claim_boundary": [
            "This is a frozen held-out readout probe on the compact LeWM latent export.",
            "Subspace predictability is descriptive and does not establish a causal role for a band.",
            "The source artifact lacks episode/timestep and physical counterfactual fields, so no online planning or physical-control claim is made.",
        ],
    }
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    rows = [
        "# Frozen LeWM covariance-subspace readout probes",
        "",
        "Covariance eigenvectors are fitted on the training split only. Each probe is a centered ridge readout evaluated on held-out rows; random-subspace values are dimension-matched null summaries.",
        "",
        "| target | subspace | k | held-out R2 | MSE | random-subspace R2 mean +/- SD |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    null = report["random_subspace_null"]
    for row in report["records"]:
        if row["subspace"] == "full":
            random_text = "--"
        else:
            key = f"{row['subspace']}{row['k']}"
            n = null[row["target"]][key]
            random_text = f"{n['r2_mean']:.5f} +/- {n['r2_std']:.5f}"
        rows.append(
            f"| {row['target']} | {row['subspace']} | {row['k']} | {row['r2']:.6f} | {row['mse']:.6g} | {random_text} |"
        )
    rows.extend(
        [
            "",
            "## Covariance diagnostics",
            "",
            f"- Training-split stable rank: `{report['covariance']['stable_rank']:.4f}`.",
            "- Top variance shares: "
            + ", ".join(
                f"k={k}: {value:.4f}" for k, value in report["covariance"]["top_variance_share"].items()
            ),
            "",
            "## Boundary",
            "",
            *[f"- {item}" for item in report["claim_boundary"]],
        ]
    )
    path.write_text("\n".join(rows) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--sample-seed", type=int, default=20261003)
    parser.add_argument("--split-seed", type=int, default=20261004)
    parser.add_argument("--ks", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--random-repeats", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()
    run_report(
        latents_path=args.latents,
        output=args.output,
        max_samples=args.max_samples,
        test_fraction=args.test_fraction,
        sample_seed=args.sample_seed,
        split_seed=args.split_seed,
        ks=args.ks,
        random_repeats=args.random_repeats,
        ridge=args.ridge,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
