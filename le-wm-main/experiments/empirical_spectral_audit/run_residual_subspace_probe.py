"""Extract prediction-residual-sensitive subspaces from frozen LeWM latents.

This is the residual-side companion to the dynamics-sensitive probe.  All
subspaces are fitted on the training split only and evaluated with held-out
ridge readouts.  In addition to a latent--residual cross-covariance basis, the
report includes a residual covariance basis and its overlap with covariance and
dynamics-sensitive geometries.

The result is a frozen representation diagnostic.  It does not establish that
training moved errors into a task-irrelevant subspace, because the source
artifact is a single frozen checkpoint rather than a matched training sweep.
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

from experiments.empirical_spectral_audit.run_dynamics_subspace_probe import (  # noqa: E402
    _cross_covariance_basis,
    _finite,
    _fit_linear_map,
    _orthonormal_subspace,
    _probe,
    _residualize_target,
)


def _covariance_basis(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, order], eigenvalues[order]


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

    permutation = np.random.default_rng(split_seed).permutation(latent.shape[0])
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
    covariance_basis, covariance_values = _covariance_basis(z_train)
    dynamics_basis, dynamics_singular_values = _cross_covariance_basis(z_train, d_train)
    residual_basis, residual_singular_values = _cross_covariance_basis(z_train, r_train)
    conditional_basis, conditional_singular_values = _cross_covariance_basis(
        z_train, conditional_d_train
    )
    residual_covariance_basis, residual_covariance_values = _covariance_basis(r_train)

    dim = int(latent.shape[1])
    ks = sorted({int(k) for k in ks})
    if any(k <= 0 or k > dim for k in ks):
        raise ValueError(f"ks must lie in [1, {dim}]")
    targets = {
        "residual": (r_train, r_test),
        "dynamics": (d_train, d_test),
        "predicted_dynamics": (p_train, p_test),
        "dynamics_after_action": (conditional_d_train, conditional_d_test),
    }
    bases = {
        "covariance": covariance_basis,
        "dynamics_cross_covariance": dynamics_basis,
        "residual_cross_covariance": residual_basis,
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
                        **_probe(z_train, z_test, target_train, target_test, basis, k, ridge),
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
                        z_train, z_test, target_train, target_test, random_basis, k, ridge
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

    def energy_profile(values: np.ndarray, basis: np.ndarray) -> dict[str, float]:
        energy = np.mean(np.square(values @ basis), axis=0)
        total = max(float(energy.sum()), 1e-12)
        return {
            "top": float(energy[: min(8, dim)].sum() / total),
            "middle": float(energy[8 : min(32, dim)].sum() / total),
            "tail": float(energy[min(128, dim) :].sum() / total),
        }

    report = {
        "artifact": "frozen_lewm_residual_sensitive_subspaces",
        "status": "frozen_residual_subspace_readout",
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
            "basis_fit": "training-split latent-to-residual cross-covariance SVD",
            "residual_covariance_basis_fit": "training-split residual covariance eigendecomposition",
            "conditional_target": "dynamics residual after training-split linear action-to-dynamics regression",
            "readout": "centered ridge regression; R2 evaluated on held-out rows",
        },
        "cross_covariance": {
            "dynamics_singular_values": dynamics_singular_values.tolist(),
            "residual_singular_values": residual_singular_values.tolist(),
            "conditional_dynamics_singular_values": conditional_singular_values.tolist(),
            "covariance_top_variance_share": {
                str(k): float(covariance_values[:k].sum() / max(covariance_values.sum(), 1e-12))
                for k in ks
            },
            "residual_covariance_top_variance_share": {
                str(k): float(residual_covariance_values[:k].sum() / max(residual_covariance_values.sum(), 1e-12))
                for k in ks
            },
            "subspace_overlap_with_covariance": {
                str(k): {
                    "dynamics_cross_covariance": overlap(covariance_basis, dynamics_basis, k),
                    "residual_cross_covariance": overlap(covariance_basis, residual_basis, k),
                    "conditional_dynamics_cross_covariance": overlap(
                        covariance_basis, conditional_basis, k
                    ),
                    "residual_covariance": overlap(covariance_basis, residual_covariance_basis, k),
                }
                for k in ks
            },
            "subspace_overlap_with_dynamics": {
                str(k): {
                    "residual_cross_covariance": overlap(dynamics_basis, residual_basis, k),
                    "residual_covariance": overlap(dynamics_basis, residual_covariance_basis, k),
                    "conditional_dynamics_cross_covariance": overlap(
                        dynamics_basis, conditional_basis, k
                    ),
                }
                for k in ks
            },
        },
        "heldout_residual_energy_profile": {
            "covariance_basis": energy_profile(r_test, covariance_basis),
            "dynamics_basis": energy_profile(r_test, dynamics_basis),
            "residual_cross_covariance_basis": energy_profile(r_test, residual_basis),
            "residual_covariance_basis": energy_profile(r_test, residual_covariance_basis),
        },
        "records": records,
        "random_subspace_null": random_null,
        "claim_boundary": [
            "Residual-sensitive bases are fit on training rows and evaluated on held-out frozen latent rows.",
            "Residual alignment and energy profiles describe where prediction error is represented; they do not establish that training intentionally moved error into a task-irrelevant subspace.",
            "The source artifact lacks episode/timestep and physical counterfactual fields, so no causal control or online planning claim is made.",
        ],
    }
    basis_path = output.with_name("residual_subspace_basis.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        basis_path,
        latent_mean=z_train.mean(axis=0),
        covariance_basis=covariance_basis,
        dynamics_basis=dynamics_basis,
        residual_basis=residual_basis,
        conditional_dynamics_basis=conditional_basis,
        residual_covariance_basis=residual_covariance_basis,
        covariance_eigenvalues=covariance_values,
        residual_covariance_eigenvalues=residual_covariance_values,
        dynamics_cross_covariance_singular_values=dynamics_singular_values,
        residual_cross_covariance_singular_values=residual_singular_values,
        conditional_dynamics_cross_covariance_singular_values=conditional_singular_values,
    )
    report["protocol"]["basis_artifact"] = str(basis_path)
    if not _finite(report):
        raise ValueError("report contains non-finite values")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, output.with_suffix(".md"))
    output.with_name("notes.txt").write_text(_notes(report))
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Residual-sensitive subspaces in frozen LeWM latents",
        "",
        "All bases are fitted on training rows only. Readout R2 and residual energy profiles are evaluated on held-out rows.",
        "",
        "| target | basis | k | held-out R2 | random R2 mean +/- SD |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in report["records"]:
        null = report["random_subspace_null"][row["target"]][str(row["k"])]
        lines.append(
            f"| {row['target']} | {row['basis']} | {row['k']} | {row['r2']:.6f} | "
            f"{null['r2_mean']:.6f} +/- {null['r2_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## k=32 overlap",
            "",
            "| comparison basis | covariance overlap | dynamics overlap |",
            "| --- | ---: | ---: |",
        ]
    )
    for basis_name in ("residual_cross_covariance", "residual_covariance", "conditional_dynamics_cross_covariance"):
        cov = report["cross_covariance"]["subspace_overlap_with_covariance"]["32"][basis_name]
        dyn = report["cross_covariance"]["subspace_overlap_with_dynamics"]["32"].get(basis_name)
        lines.append(f"| {basis_name} | {cov:.6f} | {'--' if dyn is None else f'{dyn:.6f}'} |")
    lines.extend(["", "## Held-out residual energy profile", "", "| basis | top 8 | middle 8--31 | tail 128--191 |", "| --- | ---: | ---: | ---: |"])
    for basis_name, values in report["heldout_residual_energy_profile"].items():
        lines.append(f"| {basis_name} | {values['top']:.6f} | {values['middle']:.6f} | {values['tail']:.6f} |")
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def _notes(report: dict[str, Any]) -> str:
    rows = [
        row
        for row in report["records"]
        if row["k"] == 32 and row["target"] == "residual"
    ]
    values = {row["basis"]: row["r2"] for row in rows}
    overlap = report["cross_covariance"]["subspace_overlap_with_covariance"]["32"]
    return (
        "2026-10-03: Added train-only latent--residual cross-covariance and residual covariance subspaces.\n"
        f"Held-out residual k=32 R2: residual_cross_covariance={values['residual_cross_covariance']:.6f}, covariance={values['covariance']:.6f}, dynamics_cross_covariance={values['dynamics_cross_covariance']:.6f}.\n"
        f"k=32 residual-basis overlap with covariance={overlap['residual_cross_covariance']:.6f}; residual-covariance-basis overlap with covariance={overlap['residual_covariance']:.6f}.\n"
        "Boundary: frozen residual representation diagnostic; no claim that training deliberately relocates error or improves control.\n"
    )


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
