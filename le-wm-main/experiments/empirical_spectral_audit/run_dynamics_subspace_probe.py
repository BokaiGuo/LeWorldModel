"""Extract held-out dynamics-sensitive subspaces from frozen LeWM latents.

The covariance bands are descriptive but not necessarily the best subspace for
predicting latent dynamics. This audit therefore fits two supervised latent
subspaces on the training split: a cross-covariance dynamics basis and a
conditional basis after regressing the action input out of the dynamics
target. Held-out ridge readouts compare these bases with the covariance basis
and dimension-matched random-subspace nulls.

This is a frozen representation diagnostic. It does not train LeWM or claim
that a discovered subspace causes physical control or planning success.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.run_frozen_subspace_probes import (
    _finite,
    _orthonormal_subspace,
    _ridge_probe,
)


def _fit_linear_map(
    train_features: np.ndarray,
    train_target: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(train_features, dtype=np.float64)
    y = np.asarray(train_target, dtype=np.float64)
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    xc = x - x_mean
    yc = y - y_mean
    gram = xc.T @ xc
    alpha = float(ridge) * max(float(np.trace(gram)) / max(x.shape[1], 1), 1e-12)
    coef = np.linalg.solve(gram + alpha * np.eye(x.shape[1]), xc.T @ yc)
    return coef, x_mean, y_mean


def _residualize_target(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    coef, x_mean, y_mean = _fit_linear_map(train_features, train_target, ridge)
    train_prediction = (train_features - x_mean) @ coef + y_mean
    test_prediction = (test_features - x_mean) @ coef + y_mean
    return train_target - train_prediction, test_target - test_prediction


def _cross_covariance_basis(
    z_train: np.ndarray,
    target_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z_centered = z_train - z_train.mean(axis=0, keepdims=True)
    target_centered = target_train - target_train.mean(axis=0, keepdims=True)
    cross_covariance = z_centered.T @ target_centered / max(z_train.shape[0] - 1, 1)
    left, singular_values, _ = np.linalg.svd(cross_covariance, full_matrices=False)
    return left, singular_values


def _probe(
    z_train: np.ndarray,
    z_test: np.ndarray,
    target_train: np.ndarray,
    target_test: np.ndarray,
    basis: np.ndarray,
    k: int,
    ridge: float,
) -> dict[str, float]:
    return _ridge_probe(
        z_train @ basis[:, :k],
        z_test @ basis[:, :k],
        target_train,
        target_test,
        ridge=ridge,
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
        required = {"latents", "actions", "dynamics", "predicted_dynamics", "prediction_residual"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"missing fields: {missing}")
        row_count = int(archive["latents"].shape[0])
        if max_samples <= 0 or max_samples >= row_count:
            selected = np.arange(row_count)
        else:
            selected = np.sort(
                np.random.default_rng(sample_seed).choice(
                    row_count, size=max_samples, replace=False
                )
            )
        latent = np.asarray(archive["latents"][selected], dtype=np.float64)
        actions = np.asarray(archive["actions"][selected], dtype=np.float64)
        dynamics = np.asarray(archive["dynamics"][selected], dtype=np.float64)
        predicted = np.asarray(archive["predicted_dynamics"][selected], dtype=np.float64)
        residual = np.asarray(archive["prediction_residual"][selected], dtype=np.float64)

    split_rng = np.random.default_rng(split_seed)
    permutation = split_rng.permutation(latent.shape[0])
    test_count = max(1, int(round(latent.shape[0] * test_fraction)))
    test_idx, train_idx = permutation[:test_count], permutation[test_count:]
    z_train, z_test = latent[train_idx], latent[test_idx]
    a_train, a_test = actions[train_idx], actions[test_idx]
    d_train, d_test = dynamics[train_idx], dynamics[test_idx]
    p_train, p_test = predicted[train_idx], predicted[test_idx]
    r_train, r_test = residual[train_idx], residual[test_idx]

    conditional_d_train, conditional_d_test = _residualize_target(
        a_train, a_test, d_train, d_test, ridge=ridge
    )
    covariance_centered = z_train - z_train.mean(axis=0, keepdims=True)
    covariance = covariance_centered.T @ covariance_centered / max(z_train.shape[0] - 1, 1)
    covariance_values, covariance_vectors = np.linalg.eigh(covariance)
    covariance_order = np.argsort(covariance_values)[::-1]
    covariance_values = covariance_values[covariance_order]
    covariance_basis = covariance_vectors[:, covariance_order]
    dynamics_basis, dynamics_singular_values = _cross_covariance_basis(z_train, d_train)
    conditional_basis, conditional_singular_values = _cross_covariance_basis(
        z_train, conditional_d_train
    )

    dim = int(latent.shape[1])
    ks = sorted({int(k) for k in ks})
    if any(k <= 0 or k > dim for k in ks):
        raise ValueError(f"ks must lie in [1, {dim}]")
    targets = {
        "dynamics": (d_train, d_test),
        "predicted_dynamics": (p_train, p_test),
        "prediction_residual": (r_train, r_test),
        "dynamics_after_action": (conditional_d_train, conditional_d_test),
    }
    bases = {
        "covariance": covariance_basis,
        "dynamics_cross_covariance": dynamics_basis,
        "conditional_dynamics_cross_covariance": conditional_basis,
    }
    records: list[dict[str, Any]] = []
    for target_name, (target_train, target_test) in targets.items():
        for basis_name, basis in bases.items():
            for k in ks:
                records.append(
                    {
                        "target": target_name,
                        "basis": basis_name,
                        "k": int(k),
                        **_probe(
                            z_train,
                            z_test,
                            target_train,
                            target_test,
                            basis,
                            k,
                            ridge,
                        ),
                    }
                )

    random_null: dict[str, dict[str, dict[str, float]]] = {}
    for target_name, (target_train, target_test) in targets.items():
        random_null[target_name] = {}
        for k in ks:
            values = []
            for repeat in range(random_repeats):
                random_basis = _orthonormal_subspace(
                    dim,
                    k,
                    np.random.default_rng(sample_seed + 100_000 * k + repeat),
                )
                values.append(
                    _probe(
                        z_train,
                        z_test,
                        target_train,
                        target_test,
                        random_basis,
                        k,
                        ridge,
                    )["r2"]
                )
            random_null[target_name][str(k)] = {
                "r2_mean": float(np.mean(values)),
                "r2_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "r2_min": float(np.min(values)),
                "r2_max": float(np.max(values)),
                "repeats": int(random_repeats),
            }

    def overlap(first: np.ndarray, second: np.ndarray, k: int) -> float:
        return float(np.square(first[:, :k].T @ second[:, :k]).sum() / k)

    report = {
        "artifact": "frozen_lewm_dynamics_sensitive_subspaces",
        "status": "frozen_dynamics_subspace_readout",
        "source": {
            "latents": str(latents_path),
            "source_rows": row_count,
            "selected_rows": int(latent.shape[0]),
            "latent_dim": dim,
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
            "basis_fit": "training-split latent-to-dynamics cross-covariance SVD",
            "conditional_target": "dynamics residual after training-split linear action-to-dynamics regression",
            "readout": "centered ridge regression; R2 evaluated on held-out rows",
        },
        "cross_covariance": {
            "dynamics_singular_values": dynamics_singular_values.tolist(),
            "conditional_dynamics_singular_values": conditional_singular_values.tolist(),
            "covariance_top_variance_share": {
                str(k): float(covariance_values[:k].sum() / max(covariance_values.sum(), 1e-12))
                for k in ks
            },
            "subspace_overlap_with_covariance": {
                str(k): {
                    "dynamics_cross_covariance": overlap(covariance_basis, dynamics_basis, k),
                    "conditional_dynamics_cross_covariance": overlap(
                        covariance_basis, conditional_basis, k
                    ),
                }
                for k in ks
            },
        },
        "records": records,
        "random_subspace_null": random_null,
        "claim_boundary": [
            "The supervised bases are fit on the training split and evaluated on held-out frozen latent rows.",
            "Cross-covariance directions reveal predictive dynamics information but do not establish a causal latent mechanism.",
            "The source artifact lacks episode/timestep and physical counterfactual fields, so no online planning or physical-control claim is made.",
        ],
    }
    # Create the destination before writing both the basis artifact and the
    # report.  The basis is intentionally saved first so an interrupted run
    # cannot leave a JSON report that advertises a missing coordinate system.
    output.parent.mkdir(parents=True, exist_ok=True)
    basis_path = output.with_name("dynamics_subspace_basis.npz")
    np.savez_compressed(
        basis_path,
        latent_mean=z_train.mean(axis=0),
        covariance_basis=covariance_basis,
        dynamics_basis=dynamics_basis,
        conditional_dynamics_basis=conditional_basis,
        covariance_eigenvalues=covariance_values,
        dynamics_cross_covariance_singular_values=dynamics_singular_values,
        conditional_dynamics_cross_covariance_singular_values=conditional_singular_values,
    )
    report["protocol"]["basis_artifact"] = str(basis_path)
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Dynamics-sensitive subspaces in frozen LeWM latents",
        "",
        "The dynamics cross-covariance bases are fitted on the training split only. The conditional basis uses dynamics residualized against a training-split linear action readout. All probe scores are held-out R²; random-subspace values are dimension-matched null summaries.",
        "",
        "| target | basis | k | held-out R² | random R² mean +/- SD |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    null = report["random_subspace_null"]
    for row in report["records"]:
        random_value = null[row["target"]][str(row["k"])]
        lines.append(
            f"| {row['target']} | {row['basis']} | {row['k']} | {row['r2']:.6f} | "
            f"{random_value['r2_mean']:.6f} +/- {random_value['r2_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Subspace overlap",
            "",
            "| k | covariance vs dynamics basis | covariance vs conditional dynamics basis |",
            "| ---: | ---: | ---: |",
        ]
    )
    for k, values in report["cross_covariance"]["subspace_overlap_with_covariance"].items():
        lines.append(
            f"| {k} | {values['dynamics_cross_covariance']:.6f} | "
            f"{values['conditional_dynamics_cross_covariance']:.6f} |"
        )
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--sample-seed", type=int, default=20261003)
    parser.add_argument("--split-seed", type=int, default=20261005)
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
