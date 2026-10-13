"""Cross-fitted uncertainty/residual geometry audit for frozen LeWM latents.

The existing audit establishes that dynamics, planning, nuisance, and
prediction-residual tangents are all non-random, but their planner relevance
is not determined by a single global lift.  This follow-up asks where the
largest prediction errors point.  Training-only covariance and
cross-covariance bases are transported to held-out rows.  Held-out residual
directions are then compared between low- and high-error rows, with a
within-split label permutation and dimension-matched random-basis controls.

This is a frozen, row-level diagnostic.  The source artifact has no episode
IDs, goals, success labels, or physical counterfactuals, so the result is not
a causal uncertainty estimate, active exploration result, or online planning
evaluation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


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


def _covariance_basis(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors[:, np.argsort(eigenvalues)[::-1]]


def _cross_covariance_basis(latent: np.ndarray, target: np.ndarray) -> np.ndarray:
    z = latent - latent.mean(axis=0, keepdims=True)
    y = target - target.mean(axis=0, keepdims=True)
    left, _, _ = np.linalg.svd(z.T @ y / max(latent.shape[0] - 1, 1), full_matrices=False)
    return left


def _residualize(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_mean = train_features.mean(axis=0, keepdims=True)
    y_mean = train_target.mean(axis=0, keepdims=True)
    x = train_features - x_mean
    y = train_target - y_mean
    alpha = float(ridge) * max(float(np.trace(x.T @ x)) / max(x.shape[1], 1), 1e-12)
    coef = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
    return train_target - (x @ coef + y_mean), test_target - ((test_features - x_mean) @ coef + y_mean)


def _direction_share(residual: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    norm = np.square(residual).sum(axis=1).clip(min=1e-12)
    projection = residual @ basis[:, :k]
    return np.square(projection).sum(axis=1) / norm


def _sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    exceed = 0
    for signs in itertools.product((-1.0, 1.0), repeat=int(values.size)):
        if abs(float((values * np.asarray(signs)).mean())) >= observed - 1e-15:
            exceed += 1
    return float(exceed / (2 ** int(values.size)))


def _summary(values: np.ndarray, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(10000, values.size))].mean(axis=1)
    return {
        "n_splits": int(values.size),
        "mean_high_minus_low": float(values.mean()),
        "std_high_minus_low": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "bootstrap_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "exact_two_sided_sign_flip_p": _sign_flip(values),
        "split_values": [float(value) for value in values],
    }


def _contrast_summary(values: np.ndarray, seed: int, name: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(10000, values.size))].mean(axis=1)
    return {
        "n_splits": int(values.size),
        f"mean_{name}": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "bootstrap_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "exact_two_sided_sign_flip_p": _sign_flip(values),
        "split_values": [float(value) for value in values],
    }


def _load_arrays(path: Path, max_samples: int, seed: int) -> tuple[dict[str, np.ndarray], int]:
    required = ("latents", "actions", "dynamics", "predicted_dynamics", "planning", "nuisance", "prediction_residual")
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(required) - set(archive.files))
        if missing:
            raise KeyError(f"missing fields: {missing}")
        row_count = int(archive["latents"].shape[0])
        if max_samples <= 0 or max_samples >= row_count:
            selected = np.arange(row_count)
        else:
            selected = np.sort(np.random.default_rng(seed).choice(row_count, size=max_samples, replace=False))
        arrays = {name: np.asarray(archive[name][selected], dtype=np.float64) for name in required}
    return arrays, row_count


def _one_split(
    arrays: dict[str, np.ndarray],
    seed: int,
    test_fraction: float,
    ks: list[int],
    ridge: float,
    permutation_repeats: int,
    random_basis_repeats: int,
) -> dict[str, Any]:
    n = arrays["latents"].shape[0]
    order = np.random.default_rng(seed).permutation(n)
    test_count = max(8, int(round(n * test_fraction)))
    test_idx, train_idx = order[:test_count], order[test_count:]
    train = {key: value[train_idx] for key, value in arrays.items()}
    test = {key: value[test_idx] for key, value in arrays.items()}

    conditional_train, conditional_test = _residualize(
        train["actions"], test["actions"], train["dynamics"], test["dynamics"], ridge
    )
    bases = {
        "covariance": _covariance_basis(train["latents"]),
        "dynamics": _cross_covariance_basis(train["latents"], train["dynamics"]),
        "predicted_dynamics": _cross_covariance_basis(train["latents"], train["predicted_dynamics"]),
        "planning": _cross_covariance_basis(train["latents"], train["planning"]),
        "nuisance": _cross_covariance_basis(train["latents"], train["nuisance"]),
        "conditional_dynamics": _cross_covariance_basis(train["latents"], conditional_train),
        "residual_covariance": _covariance_basis(train["prediction_residual"]),
    }
    residual = test["prediction_residual"]
    score_train = np.linalg.norm(train["prediction_residual"], axis=1)
    score_test = np.linalg.norm(residual, axis=1)
    low_threshold = float(np.quantile(score_train, 0.25))
    high_threshold = float(np.quantile(score_train, 0.75))
    low_mask = score_test <= low_threshold
    high_mask = score_test >= high_threshold
    if int(low_mask.sum()) < 2 or int(high_mask.sum()) < 2:
        raise RuntimeError("uncertainty quantile split has too few held-out rows")

    records: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 100_000)
    for basis_name, basis in bases.items():
        for k in ks:
            share = _direction_share(residual, basis, k)
            observed = float(share[high_mask].mean() - share[low_mask].mean())
            null_values = []
            labels = np.concatenate((np.ones(int(high_mask.sum()), dtype=bool), np.zeros(int(low_mask.sum()), dtype=bool)))
            values = np.concatenate((share[high_mask], share[low_mask]))
            for _ in range(int(permutation_repeats)):
                shuffled = rng.permutation(labels)
                null_values.append(float(values[shuffled].mean() - values[~shuffled].mean()))
            records.append({
                "basis": basis_name,
                "k": int(k),
                "high_mean_share": float(share[high_mask].mean()),
                "low_mean_share": float(share[low_mask].mean()),
                "high_minus_low": observed,
                "permutation_null_mean": float(np.mean(null_values)),
                "permutation_null_std": float(np.std(null_values, ddof=1)),
                "permutation_null_q025": float(np.quantile(null_values, 0.025)),
                "permutation_null_q975": float(np.quantile(null_values, 0.975)),
                "n_high": int(high_mask.sum()),
                "n_low": int(low_mask.sum()),
            })

    random_null: dict[str, dict[str, float]] = {}
    dim = int(residual.shape[1])
    for k in ks:
        values = []
        for repeat in range(int(random_basis_repeats)):
            random_basis, _ = np.linalg.qr(rng.normal(size=(dim, k)))
            share = _direction_share(residual, random_basis, k)
            values.append(float(share[high_mask].mean() - share[low_mask].mean()))
        random_null[str(k)] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
            "repeats": int(random_basis_repeats),
        }
    return {
        "split_seed": int(seed),
        "train_rows": int(train_idx.size),
        "test_rows": int(test_idx.size),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "train_score_mean": float(score_train.mean()),
        "test_score_mean": float(score_test.mean()),
        "records": records,
        "random_basis_null": random_null,
    }


def run_report(
    latents_path: Path,
    output: Path,
    max_samples: int,
    seeds: list[int],
    test_fraction: float,
    ks: list[int],
    ridge: float,
    permutation_repeats: int,
    random_basis_repeats: int,
) -> dict[str, Any]:
    started = time.time()
    split_reports = []
    row_count = None
    for seed in seeds:
        arrays, row_count = _load_arrays(latents_path, max_samples, int(seed) + 70_000)
        split_reports.append(_one_split(arrays, int(seed), test_fraction, ks, ridge, permutation_repeats, random_basis_repeats))

    summary: dict[str, Any] = {}
    for basis in ("covariance", "dynamics", "predicted_dynamics", "planning", "nuisance", "conditional_dynamics", "residual_covariance"):
        summary[basis] = {}
        for k in ks:
            values = np.asarray([row["high_minus_low"] for report in split_reports for row in report["records"] if row["basis"] == basis and row["k"] == k], dtype=np.float64)
            summary[basis][str(k)] = _summary(values, seed=20261101 + int(k))

    basis_contrasts: dict[str, Any] = {}
    for first, second in (
        ("nuisance", "planning"),
        ("covariance", "planning"),
        ("residual_covariance", "planning"),
        ("dynamics", "planning"),
        ("conditional_dynamics", "planning"),
    ):
        basis_contrasts[f"{first}_minus_{second}"] = {}
        for k in ks:
            values = []
            for split in split_reports:
                lookup = {(row["basis"], row["k"]): row["high_minus_low"] for row in split["records"]}
                values.append(lookup[(first, int(k))] - lookup[(second, int(k))])
            basis_contrasts[f"{first}_minus_{second}"][str(k)] = {
                "basis_first": first,
                "basis_second": second,
                "k": int(k),
                **_contrast_summary(np.asarray(values), seed=20261201 + int(k), name=f"{first}_minus_{second}"),
            }

    interpretation = [
        "High-minus-low values are held-out differences in normalized residual-direction energy, with thresholds calibrated only on each training split.",
        "A positive value means high-error rows point more strongly into the named basis than low-error rows; it is not a causal effect.",
        "The random-basis and uncertainty-label permutation controls separate orientation from residual magnitude and from arbitrary subspaces.",
    ]
    focus_k = 8 if 8 in ks else max(ks)
    nuisance_planning_k8 = basis_contrasts["nuisance_minus_planning"][str(focus_k)]
    covariance_planning_k8 = basis_contrasts["covariance_minus_planning"][str(focus_k)]
    interpretation.extend([
        f"At k={focus_k}, the nuisance-minus-planning high-minus-low contrast is {nuisance_planning_k8['mean_nuisance_minus_planning']:.6g} with split-level sign-flip p={nuisance_planning_k8['exact_two_sided_sign_flip_p']:.6g}.",
        f"At k={focus_k}, the covariance-minus-planning contrast is {covariance_planning_k8['mean_covariance_minus_planning']:.6g}; this supports a descriptive error-geometry separation, not a causal nuisance attribution.",
    ])
    report = {
        "artifact": "frozen_lewm_uncertainty_subspace_conflict_audit",
        "status": "complete_cross_fitted_residual_direction_audit",
        "source": {"latents": str(latents_path), "source_rows": int(row_count or 0), "selected_rows_per_split": int(max_samples)},
        "protocol": {
            "seeds": [int(seed) for seed in seeds],
            "test_fraction": float(test_fraction),
            "ks": [int(k) for k in ks],
            "ridge": float(ridge),
            "uncertainty_score": "Euclidean norm of frozen prediction_residual",
            "thresholds": "training-split 25th/75th percentiles; held-out middle rows excluded",
            "basis_fit": "training-split covariance eigenspaces and latent-to-tangent cross-covariance SVD",
            "action_residualization": "training-split ridge regression of dynamics on actions",
            "permutation_repeats_per_cell": int(permutation_repeats),
            "random_basis_repeats_per_k": int(random_basis_repeats),
            "row_level_split_without_episode_ids": True,
            "synthetic_or_frozen": "frozen LeWM latent artifact",
            "elapsed_seconds": time.time() - started,
        },
        "split_reports": split_reports,
        "summary": summary,
        "basis_contrasts": basis_contrasts,
        "interpretation": interpretation,
        "claim_boundary": [
            "This is a frozen, row-level representation diagnostic, not an online uncertainty or active-exploration evaluation.",
            "The held-out residual norm is used as an error/uncertainty proxy; it is not an epistemic uncertainty estimator without repeated models or stochastic inference.",
            "The artifact lacks episode IDs, goals, success labels, physical variables, and counterfactual transitions, so no causal control or task-success claim is made.",
        ],
    }
    _finite(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, output.with_suffix(".md"))
    return report


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Frozen LeWM uncertainty-subspace conflict audit",
        "",
        "Training-split bases are evaluated on held-out rows. High and low uncertainty groups use training-only residual-norm thresholds; the middle group is excluded. Positive high-minus-low values mean that high-error residual directions have greater normalized energy in the named basis.",
        "",
        "| basis | k | n splits | mean high-low | 95% bootstrap CI | sign-flip p |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for basis, cells in report["summary"].items():
        for k, value in cells.items():
            ci = value["bootstrap_ci95"]
            lines.append(f"| {basis} | {k} | {value['n_splits']} | {value['mean_high_minus_low']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] | {value['exact_two_sided_sign_flip_p']:.6g} |")
    lines.extend(["", "## Paired basis contrasts", "", "These contrasts use the same held-out high/low rows within each split.", "", "| first basis - second basis | k | n splits | mean contrast | 95% bootstrap CI | sign-flip p |", "| --- | ---: | ---: | ---: | --- | ---: |"])
    for name, cells in report["basis_contrasts"].items():
        for k, value in cells.items():
            ci = value["bootstrap_ci95"]
            lines.append(f"| {name} | {k} | {value['n_splits']} | {value[f'mean_{name}']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] | {value['exact_two_sided_sign_flip_p']:.6g} |")
    lines.extend(["", "## Interpretation", "", *[f"- {item}" for item in report["interpretation"]], "", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=60000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3072, 3073, 3074])
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--permutation-repeats", type=int, default=500)
    parser.add_argument("--random-basis-repeats", type=int, default=64)
    args = parser.parse_args()
    print("Running cross-fitted frozen residual-direction audit: dynamics, planning, nuisance, and random controls.", flush=True)
    report = run_report(args.latents, args.output, args.max_samples, args.seeds, args.test_fraction, args.ks, args.ridge, args.permutation_repeats, args.random_basis_repeats)
    print(f"wrote {args.output} ({len(report['split_reports'])} splits)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
