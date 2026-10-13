"""Gauge and non-normality audit for covariance--predictor diagnostics.

The audit addresses two limitations of a Euclidean commutator:

* under a general invertible latent reparameterization ``z' = A z``, the
  covariance transforms by congruence while a linear predictor transforms by
  similarity, so the ordinary Frobenius commutator norm is not intrinsic;
* learned predictors need not be symmetric or normal, so eigenvector angles
  alone are not a complete description.

The metric-aware construction treats a latent metric M as part of the declared
operator pair.  With covariance ``Sigma`` represented as a contravariant
matrix, ``S = Sigma @ M`` is an endomorphism.  Under
``M' = A^{-T} M A^{-1}``, both ``S`` and ``P`` transform by similarity, and
the M-weighted Frobenius norm is invariant.

The script also verifies the arbitrary-predictor identity

    ||[Sigma, P]||_F^2 = sum_ij (sigma_i-sigma_j)^2 |P_tilde_ij|^2,

where P_tilde is P in the covariance eigenbasis, and reports a spectrum-fixed
upper-triangular Schur family whose non-normality changes while eigenvalues are
held fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def metric_norm(operator: np.ndarray, metric: np.ndarray) -> float:
    value = np.trace(operator.T @ metric @ operator @ np.linalg.inv(metric))
    return float(np.sqrt(max(float(value), 0.0)))


def metric_commutator(sigma: np.ndarray, predictor: np.ndarray, metric: np.ndarray) -> float:
    covariance_operator = sigma @ metric
    commutator = covariance_operator @ predictor - predictor @ covariance_operator
    denominator = metric_norm(covariance_operator, metric) * metric_norm(predictor, metric)
    return metric_norm(commutator, metric) / max(denominator, 1e-12)


def euclidean_commutator(sigma: np.ndarray, predictor: np.ndarray) -> float:
    commutator = sigma @ predictor - predictor @ sigma
    denominator = np.linalg.norm(sigma, "fro") * np.linalg.norm(predictor, "fro")
    return float(np.linalg.norm(commutator, "fro") / max(denominator, 1e-12))


def nonnormality(predictor: np.ndarray) -> float:
    numerator = np.linalg.norm(predictor.T @ predictor - predictor @ predictor.T, "fro")
    denominator = np.linalg.norm(predictor, "fro") ** 2
    return float(numerator / max(denominator, 1e-12))


def covariance_basis_identity(sigma: np.ndarray, predictor: np.ndarray) -> tuple[float, float]:
    values, vectors = np.linalg.eigh(sigma)
    transformed = vectors.T @ predictor @ vectors
    lhs = np.linalg.norm(sigma @ predictor - predictor @ sigma, "fro") ** 2
    rhs = float(np.sum((values[:, None] - values[None, :]) ** 2 * np.abs(transformed) ** 2))
    return float(lhs), rhs


def shear(dimension: int, strength: float) -> np.ndarray:
    matrix = np.eye(dimension)
    matrix[0, 1] = strength
    matrix[1, 2] = -0.5 * strength
    return matrix


def rotation(dimension: int, i: int, j: int, degrees: float) -> np.ndarray:
    matrix = np.eye(dimension)
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    matrix[i, i] = matrix[j, j] = c
    matrix[i, j] = -s
    matrix[j, i] = s
    return matrix


def run(seed: int = 20261008) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    sigma = np.diag([9.0, 4.0, 1.5, 0.5])
    lambda_diag = np.diag([0.9, 0.65, 0.4, 0.2])
    predictor = rotation(4, 0, 2, 37.0) @ lambda_diag @ rotation(4, 0, 2, 37.0).T
    metric = np.diag([1.0, 1.7, 0.8, 2.0])
    transform = shear(4, 1.3) @ rotation(4, 1, 3, 23.0)
    transformed_sigma = transform @ sigma @ transform.T
    transformed_predictor = transform @ predictor @ np.linalg.inv(transform)
    transformed_metric = np.linalg.inv(transform).T @ metric @ np.linalg.inv(transform)

    identity_lhs, identity_rhs = covariance_basis_identity(sigma, predictor)
    orthogonal = rotation(4, 0, 1, 29.0)
    orthogonal_naive_delta = abs(
        euclidean_commutator(sigma, predictor)
        - euclidean_commutator(orthogonal @ sigma @ orthogonal.T, orthogonal @ predictor @ orthogonal.T)
    )

    nonnormal_rows = []
    for alpha in (0.0, 0.25, 0.5, 1.0):
        schur = lambda_diag.copy()
        schur[0, 1] = alpha
        schur[1, 2] = -0.4 * alpha
        schur[2, 3] = 0.25 * alpha
        candidate = rotation(4, 0, 2, 37.0) @ schur @ rotation(4, 0, 2, 37.0).T
        nonnormal_rows.append(
            {
                "alpha": alpha,
                "eigenvalues": [
                    [float(value.real), float(value.imag)]
                    for value in np.sort_complex(np.linalg.eigvals(candidate))
                ],
                "nonnormality": nonnormality(candidate),
                "euclidean_commutator": euclidean_commutator(sigma, candidate),
                "metric_commutator": metric_commutator(sigma, candidate, metric),
            }
        )

    return {
        "protocol": {
            "coordinate_transform": "general invertible shear times orthogonal rotation",
            "metric_transform": "M_prime = A^{-T} M A^{-1}",
            "spectrum_fixed_nonnormal_family": True,
            "claim_boundary": "gauge and non-normality validation; not a learned-world-model or planning result",
        },
        "gauge": {
            "original_euclidean_commutator": euclidean_commutator(sigma, predictor),
            "reparameterized_euclidean_commutator": euclidean_commutator(transformed_sigma, transformed_predictor),
            "original_metric_commutator": metric_commutator(sigma, predictor, metric),
            "reparameterized_metric_commutator": metric_commutator(transformed_sigma, transformed_predictor, transformed_metric),
            "orthogonal_euclidean_invariance_absolute_delta": orthogonal_naive_delta,
        },
        "arbitrary_predictor_identity": {
            "left": identity_lhs,
            "right": identity_rhs,
            "absolute_delta": abs(identity_lhs - identity_rhs),
        },
        "spectrum_fixed_nonnormal_family": nonnormal_rows,
        "random_seed": seed,
        "random_probe": rng.normal(size=3).tolist(),
    }


def markdown(report: dict[str, object]) -> str:
    gauge = report["gauge"]
    identity = report["arbitrary_predictor_identity"]
    lines = [
        "# Gauge and non-normality audit",
        "",
        "This audit validates the metric-aware covariance--predictor diagnostic and",
        "the arbitrary-predictor covariance-basis identity. It is not a learned-world-model",
        "or planning result.",
        "",
        "## Gauge transformation",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Euclidean commutator before reparameterization | {gauge['original_euclidean_commutator']:.9g} |",
        f"| Euclidean commutator after general invertible reparameterization | {gauge['reparameterized_euclidean_commutator']:.9g} |",
        f"| Metric-aware commutator before reparameterization | {gauge['original_metric_commutator']:.9g} |",
        f"| Metric-aware commutator after reparameterization | {gauge['reparameterized_metric_commutator']:.9g} |",
        f"| Orthogonal Euclidean invariance absolute error | {gauge['orthogonal_euclidean_invariance_absolute_delta']:.3g} |",
        "",
        "The ordinary Euclidean statistic is gauge-dependent under a general",
        "invertible latent change of coordinates, while the metric-aware statistic",
        "is invariant when the latent metric is transformed with the coordinates.",
        "",
        "## Arbitrary-predictor identity",
        "",
        f"The covariance-basis identity has absolute numerical error {identity['absolute_delta']:.3g}.",
        "",
        "## Non-normal family",
        "",
        "| Schur off-diagonal strength | Non-normality | Metric-aware commutator |",
        "|---:|---:|---:|",
    ]
    for row in report["spectrum_fixed_nonnormal_family"]:
        lines.append(f"| {row['alpha']:.2f} | {row['nonnormality']:.6g} | {row['metric_commutator']:.6g} |")
    lines.extend(
        [
            "",
            "All rows share the same eigenvalue list; Schur non-normality changes independently.",
            "The appropriate learned-predictor report therefore includes metric-aware",
            "alignment together with non-normality and Schur-subspace geometry, rather",
            "than treating eigenvector angles as a complete description.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".md").write_text(markdown(report))
    print(json.dumps(report["gauge"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
