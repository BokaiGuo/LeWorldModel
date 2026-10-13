"""Audit the gauge-aware, non-normal, and nonlinear transfer extension.

The audit has three purposes:

1. verify that a metric-aware covariance--predictor diagnostic is invariant
   under a general invertible latent reparameterization;
2. show that covariance alone cannot provide a nontrivial full-GL alignment
   statistic, because covariance whitening makes the covariance endomorphism
   the identity;
3. verify a Schur/resolvent finite-horizon readout and a local nonlinear
   Taylor remainder bound for a Jacobian product.

This is a mathematical audit and synthetic control, not learned-model or
planning evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import schur


def spd_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=np.float64))
    if np.any(values <= 0.0):
        raise ValueError("metric must be positive definite")
    return (vectors * np.sqrt(values)) @ vectors.T


def metric_whitened_pair(
    covariance: np.ndarray, predictor: np.ndarray, metric: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    root = spd_sqrt(metric)
    inverse_root = np.linalg.inv(root)
    covariance_endomorphism = root @ covariance @ root
    predictor_endomorphism = root @ predictor @ inverse_root
    return covariance_endomorphism, predictor_endomorphism


def frobenius_metric(operator: np.ndarray, metric: np.ndarray) -> float:
    value = np.trace(operator.T @ metric @ operator @ np.linalg.inv(metric))
    return float(np.sqrt(max(float(value), 0.0)))


def metric_commutator(covariance: np.ndarray, predictor: np.ndarray, metric: np.ndarray) -> float:
    covariance_endomorphism, predictor_endomorphism = metric_whitened_pair(covariance, predictor, metric)
    commutator = covariance_endomorphism @ predictor_endomorphism - predictor_endomorphism @ covariance_endomorphism
    denominator = np.linalg.norm(covariance_endomorphism, "fro") * np.linalg.norm(predictor_endomorphism, "fro")
    return float(np.linalg.norm(commutator, "fro") / max(denominator, 1e-12))


def euclidean_commutator(covariance: np.ndarray, predictor: np.ndarray) -> float:
    commutator = covariance @ predictor - predictor @ covariance
    denominator = np.linalg.norm(covariance, "fro") * np.linalg.norm(predictor, "fro")
    return float(np.linalg.norm(commutator, "fro") / max(denominator, 1e-12))


def metric_nonnormality(predictor: np.ndarray, metric: np.ndarray) -> float:
    _, predictor_endomorphism = metric_whitened_pair(np.eye(predictor.shape[0]), predictor, metric)
    numerator = np.linalg.norm(
        predictor_endomorphism.T @ predictor_endomorphism
        - predictor_endomorphism @ predictor_endomorphism.T,
        "fro",
    )
    return float(numerator / max(np.linalg.norm(predictor_endomorphism, "fro") ** 2, 1e-12))


def covariance_only_whitening_no_go(covariance: np.ndarray, predictor: np.ndarray) -> dict[str, float]:
    inverse_covariance = np.linalg.inv(covariance)
    covariance_endomorphism, _ = metric_whitened_pair(covariance, predictor, inverse_covariance)
    return {
        "whitened_covariance_endomorphism_error": float(np.linalg.norm(covariance_endomorphism - np.eye(covariance.shape[0]), "fro")),
        "covariance_only_commutator": metric_commutator(covariance, predictor, inverse_covariance),
    }


def nonlinear_taylor_audit(predictor: np.ndarray, metric: np.ndarray) -> dict[str, float]:
    """Check the two-step Jacobian remainder bound for f(x)=Px+beta sin(x)."""
    root = spd_sqrt(metric)
    inverse_root = np.linalg.inv(root)

    def norm_m(vector: np.ndarray) -> float:
        return float(np.linalg.norm(root @ vector))

    beta = 0.12
    base = np.asarray([0.08, -0.05, 0.04, 0.02], dtype=np.float64)
    delta = np.asarray([0.002, -0.001, 0.0015, -0.001], dtype=np.float64)

    def function(value: np.ndarray) -> np.ndarray:
        return predictor @ value + beta * np.sin(value)

    def jacobian(value: np.ndarray) -> np.ndarray:
        return predictor + beta * np.diag(np.cos(value))

    base_states = [base]
    for _ in range(2):
        base_states.append(function(base_states[-1]))
    perturbed = base + delta
    perturbed_states = [perturbed]
    for _ in range(2):
        perturbed_states.append(function(perturbed_states[-1]))

    jacobians = [jacobian(value) for value in base_states[:-1]]
    linear_product = jacobians[1] @ jacobians[0] @ delta
    actual_difference = perturbed_states[-1] - base_states[-1]
    remainder = norm_m(actual_difference - linear_product)

    # The sine Jacobian is beta-Lipschitz in Euclidean norm.  We use a
    # conservative metric-compatible tube constant in the bound.
    metric_condition = np.linalg.cond(root)
    hessian_bound = beta * metric_condition
    local_lipschitz = max(
        np.linalg.norm(root @ jacobian(value) @ inverse_root, 2)
        for value in (*base_states, *perturbed_states)
    )
    delta_norm = norm_m(delta)
    bound = 0.5 * hessian_bound * delta_norm**2 * (
        local_lipschitz**2 * local_lipschitz + local_lipschitz
    )
    return {
        "actual_two_step_linearization_remainder": remainder,
        "certified_two_step_remainder_bound": float(bound),
        "bound_slack": float(bound - remainder),
        "delta_metric_norm": delta_norm,
        "local_lipschitz_bound": float(local_lipschitz),
        "jacobian_lipschitz_bound": float(hessian_bound),
    }


def jordan_zero_commutator_audit() -> dict[str, object]:
    """Audit a fixed-spectrum Jordan family with identically zero commutator."""
    dimension = 2
    radius = 0.9
    covariance = np.eye(dimension)
    metric = np.eye(dimension)
    horizon = 8
    reference_eigenvalues = np.full(dimension, radius, dtype=np.complex128)
    rows = []
    for kappa in (0.0, 0.25, 0.5, 1.0):
        predictor = np.asarray(
            [[radius, kappa], [0.0, radius]], dtype=np.float64
        )
        gains = [
            float(np.linalg.norm(np.linalg.matrix_power(predictor, h), 2))
            for h in range(1, horizon + 1)
        ]
        power = np.linalg.matrix_power(predictor, horizon)
        rows.append(
            {
                "kappa": kappa,
                "eigenvalue_match_max_abs": float(
                    np.max(
                        np.abs(
                            np.sort_complex(np.linalg.eigvals(predictor))
                            - reference_eigenvalues
                        )
                    )
                ),
                "covariance_commutator": metric_commutator(covariance, predictor, metric),
                "metric_nonnormality": metric_nonnormality(predictor, metric),
                "finite_horizon_gain_h8": gains[-1],
                "max_gain_h1_to_h8": max(gains),
                "max_gain_horizon": int(np.argmax(gains) + 1),
                "covariance_weighted_energy_h8": float(np.trace(power @ power.T)),
            }
        )
    return {
        "covariance": "identity",
        "predictor_family": "[[r, kappa], [0, r]] with r=0.9",
        "horizon": horizon,
        "rows": rows,
    }


def run(seed: int = 20261008) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    dimension = 4
    covariance = np.diag([9.0, 4.0, 1.5, 0.5])
    metric = np.diag([1.0, 1.7, 0.8, 2.0])
    rotation = np.eye(dimension)
    theta = np.deg2rad(31.0)
    rotation[0, 0] = rotation[2, 2] = np.cos(theta)
    rotation[0, 2] = -np.sin(theta)
    rotation[2, 0] = np.sin(theta)
    diagonal = np.diag([0.9, 0.65, 0.4, 0.2])
    schur_base = diagonal.copy()
    schur_base[0, 1] = 0.7
    schur_base[1, 2] = -0.35
    predictor = rotation @ schur_base @ rotation.T

    transform = np.asarray(
        [[1.0, 1.3, 0.0, 0.0], [0.0, 1.0, -0.4, 0.0], [0.0, 0.0, 1.0, 0.8], [0.2, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transformed_covariance = transform @ covariance @ transform.T
    transformed_predictor = transform @ predictor @ np.linalg.inv(transform)
    transformed_metric = np.linalg.inv(transform).T @ metric @ np.linalg.inv(transform)

    covariance_endomorphism, predictor_endomorphism = metric_whitened_pair(covariance, predictor, metric)
    transformed_covariance_endomorphism, transformed_predictor_endomorphism = metric_whitened_pair(
        transformed_covariance, transformed_predictor, transformed_metric
    )
    gauge_whitened_predictor_error = float(
        np.linalg.norm(
            np.linalg.eigvalsh(predictor_endomorphism.T @ predictor_endomorphism)
            - np.linalg.eigvalsh(transformed_predictor_endomorphism.T @ transformed_predictor_endomorphism)
        )
    )

    alpha_rows = []
    reference_eigenvalues = np.sort_complex(np.linalg.eigvals(diagonal))
    metric_root = spd_sqrt(metric)
    metric_inverse_root = np.linalg.inv(metric_root)
    for alpha in (0.0, 0.25, 0.5, 1.0):
        triangular = diagonal.copy()
        triangular[0, 1] = 0.7 * alpha
        triangular[1, 2] = -0.35 * alpha
        triangular[2, 3] = 0.2 * alpha
        # Construct the family in metric-whitened coordinates, then pull it
        # back.  This makes alpha=0 normal in the declared metric while
        # preserving the same eigenvalue multiset for every alpha.
        whitened_candidate = rotation @ triangular @ rotation.T
        candidate = metric_inverse_root @ whitened_candidate @ metric_root
        whitened = metric_whitened_pair(np.eye(dimension), candidate, metric)[1]
        schur_form = schur(whitened, output="real")[1]
        gain_h8 = np.linalg.norm(np.linalg.matrix_power(whitened, 8), 2)
        resolvent_grid = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
        resolvent = max(
            np.linalg.norm(np.linalg.inv(np.exp(1j * angle) * np.eye(dimension) - whitened), 2)
            for angle in resolvent_grid
        )
        # A circle |zeta|=1.2 encloses the spectrum (the largest eigenvalue
        # is 0.9).  Its discretized resolvent maximum gives the explicit
        # Cauchy-circle upper bound r^(h+1) max||R(zeta)|| for h=8.
        cauchy_circle_radius = 1.2
        cauchy_circle_bound_h8 = cauchy_circle_radius**9 * max(
            np.linalg.norm(
                np.linalg.inv(
                    cauchy_circle_radius * np.exp(1j * angle) * np.eye(dimension)
                    - whitened
                ),
                2,
            )
            for angle in resolvent_grid
        )
        alpha_rows.append(
            {
                "alpha": alpha,
                "eigenvalue_match_max_abs": float(np.max(np.abs(np.sort_complex(np.linalg.eigvals(candidate)) - reference_eigenvalues))),
                "metric_nonnormality": metric_nonnormality(candidate, metric),
                "finite_horizon_gain_h8": float(gain_h8),
                "unit_circle_resolvent_max": float(resolvent),
                "cauchy_circle_bound_h8": float(cauchy_circle_bound_h8),
                "schur_off_diagonal_norm": float(np.linalg.norm(schur_form - np.diag(np.diag(schur_form)), "fro")),
            }
        )

    return {
        "protocol": {
            "coordinate_group": "general GL(d) reparameterization with a declared SPD metric",
            "nonlinear_family": "two-step f(x)=P x + beta sin(x)",
            "claim_boundary": "mathematical and synthetic audit; not learned-model or planning evidence",
        },
        "gauge": {
            "euclidean_before": euclidean_commutator(covariance, predictor),
            "euclidean_after": euclidean_commutator(transformed_covariance, transformed_predictor),
            "metric_before": metric_commutator(covariance, predictor, metric),
            "metric_after": metric_commutator(transformed_covariance, transformed_predictor, transformed_metric),
            "metric_invariance_abs_error": abs(
                metric_commutator(covariance, predictor, metric)
                - metric_commutator(transformed_covariance, transformed_predictor, transformed_metric)
            ),
            "whitened_predictor_singular_value_error": gauge_whitened_predictor_error,
        },
        "covariance_only_no_go": covariance_only_whitening_no_go(covariance, predictor),
        "spectrum_fixed_nonnormal_family": alpha_rows,
        "jordan_zero_commutator_control": jordan_zero_commutator_audit(),
        "nonlinear_jacobian_bound": nonlinear_taylor_audit(predictor, metric),
        "random_probe": rng.normal(size=3).tolist(),
    }


def markdown(report: dict[str, object]) -> str:
    gauge = report["gauge"]
    no_go = report["covariance_only_no_go"]
    nonlinear = report["nonlinear_jacobian_bound"]
    jordan = report["jordan_zero_commutator_control"]
    lines = [
        "# Intrinsic, non-normal, and nonlinear transfer audit",
        "",
        "The audit separates three facts that a spectrum-only theorem conflates:",
        "(i) full-GL gauge dependence without an external metric, (ii) non-normal transient growth at fixed eigenvalues, and (iii) nonlinear Taylor remainder beyond a local Jacobian.",
        "",
        "## Gauge",
        "",
        f"Euclidean commutator before/after a general invertible reparameterization: {gauge['euclidean_before']:.8g} / {gauge['euclidean_after']:.8g}.",
        f"Metric-aware commutator before/after: {gauge['metric_before']:.8g} / {gauge['metric_after']:.8g}; absolute invariance error {gauge['metric_invariance_abs_error']:.3g}.",
        f"Covariance-only whitening produces covariance endomorphism error {no_go['whitened_covariance_endomorphism_error']:.3g} and commutator {no_go['covariance_only_commutator']:.3g}.",
        "Therefore a nontrivial full-GL alignment statistic requires a declared additional metric, such as a decoder pullback, observation metric, or action/goal metric.",
        "",
        "## Fixed-spectrum non-normal family",
        "",
        "| Schur strength | eigenvalue error | metric non-normality | h=8 gain | unit-circle resolvent | Cauchy bound h=8 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["spectrum_fixed_nonnormal_family"]:
        lines.append(
            f"| {row['alpha']:.2f} | {row['eigenvalue_match_max_abs']:.3g} | {row['metric_nonnormality']:.6g} | {row['finite_horizon_gain_h8']:.6g} | {row['unit_circle_resolvent_max']:.6g} | {row['cauchy_circle_bound_h8']:.6g} |"
        )
    lines.extend(
        [
            "",
            "The eigenvalue multiset is fixed while Schur off-diagonal coupling, resolvent sensitivity, and finite-horizon gain vary; the Cauchy-circle column is an explicit upper bound on the measured h=8 gain.",
            "",
            "## Zero-commutator Jordan control",
            "",
            "This control sets the covariance to the identity and uses [[r, kappa], [0, r]] with fixed r=0.9. The covariance commutator is zero for every kappa, but the repeated-eigenvalue Jordan coupling changes the finite-horizon gain and covariance-weighted energy:",
            "",
            "| kappa | eigenvalue error | commutator | non-normality | h=8 gain | max gain (h=1..8) | peak h | h=8 energy |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in jordan["rows"]:
        lines.append(
            f"| {row['kappa']:.2f} | {row['eigenvalue_match_max_abs']:.3g} | {row['covariance_commutator']:.3g} | {row['metric_nonnormality']:.6g} | {row['finite_horizon_gain_h8']:.6g} | {row['max_gain_h1_to_h8']:.6g} | {row['max_gain_horizon']} | {row['covariance_weighted_energy_h8']:.6g} |"
        )
    lines.extend(
        [
            "",
            "The Jordan control is a counterexample to interpreting a small covariance commutator as a no-transient-growth certificate: the spectrum and commutator remain fixed while the nilpotent coupling changes powers. It is synthetic evidence for separating alignment, Schur geometry, and finite-horizon amplification, not learned-model or planning evidence.",
            "",
            "## Nonlinear local bound",
            "",
            f"Two-step Jacobian linearization remainder: {nonlinear['actual_two_step_linearization_remainder']:.6g}; certified bound: {nonlinear['certified_two_step_remainder_bound']:.6g}; slack: {nonlinear['bound_slack']:.6g}.",
            "The nonlinear certificate must therefore report local Jacobians, a Jacobian-Lipschitz/remainder estimate, and the finite-horizon Schur/resolvent term; replacing a nonlinear predictor by one global eigenspectrum is insufficient.",
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
    print(json.dumps({"gauge": report["gauge"], "nonlinear": report["nonlinear_jacobian_bound"]}, indent=2))


if __name__ == "__main__":
    main()
