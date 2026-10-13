"""Cross-fitted controls for learned temporal latent operators.

The checkpoint evaluators fit a linear map from target latents to predicted
latents.  A same-sample, full-rank fit is deliberately treated as descriptive:
with 200 episodes and 384 features it is not an identified operator.  This
script tests the lower-rank object out of sample and compares it with two direct
alternatives:

1. pair-shuffled training targets, which destroy episode correspondence; and
2. spectrum-preserving random-basis rotations of the paired fitted operator.

All centering, PCA, ridge fitting, and covariance construction use training
folds only.  Held-out folds are used only for prediction and commutator scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="Checkpoint metric JSON containing operator_fit_vectors; repeat for each model.")
    parser.add_argument("--label", action="append",
                        help="Reader-facing model label corresponding to each --input.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--ridge", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--control-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--primary-rank", type=int, default=16)
    parser.add_argument("--primary-ridge", type=float, default=1e-2)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | int]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=True)) if len(values) > 1 else 0.0,
        "q05": float(torch.quantile(tensor, 0.05)),
        "q95": float(torch.quantile(tensor, 0.95)),
        "n": len(values),
    }


def score(x: torch.Tensor, y: torch.Tensor, operator: torch.Tensor) -> tuple[float, float, float]:
    predicted = x @ operator
    residual = torch.sum((predicted - y) ** 2)
    baseline = torch.sum(y ** 2).clamp_min(1e-12)
    r2 = 1.0 - residual / baseline
    relative_mse = residual / baseline
    covariance = (x.T @ x) / max(1, len(x) - 1)
    numerator = torch.linalg.matrix_norm(covariance @ operator - operator @ covariance)
    denominator = torch.linalg.matrix_norm(covariance @ operator).clamp_min(1e-12)
    return float(r2), float(relative_mse), float(numerator / denominator)


def orthogonal(rank: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(rank, rank, generator=generator, dtype=torch.float64))
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    return q * signs


def fit_ridge(x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    gram = x.T @ x
    scale = torch.trace(gram).clamp_min(1e-12) / gram.shape[0]
    penalty = ridge * scale
    return torch.linalg.solve(
        gram + penalty * torch.eye(gram.shape[0], dtype=gram.dtype),
        x.T @ y,
    )


def fold_indices(n: int, folds: int, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    permutation = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    chunks = torch.tensor_split(permutation, folds)
    result = []
    for fold, test in enumerate(chunks):
        train = torch.cat([part for index, part in enumerate(chunks) if index != fold])
        result.append((train, test))
    return result


def analyze_horizon(
    target: torch.Tensor,
    predicted: torch.Tensor,
    args: argparse.Namespace,
) -> dict:
    records: dict[str, dict[str, list[float]]] = {}
    for rank in args.ranks:
        for ridge in args.ridge:
            key = f"rank={rank},ridge={ridge:g}"
            records[key] = {
                "paired_r2": [], "paired_relative_mse": [], "paired_commutator": [],
                "identity_r2": [], "zero_r2": [],
                "shuffled_r2": [], "shuffled_relative_mse": [], "shuffled_commutator": [],
                "spectral_null_r2": [], "spectral_null_relative_mse": [], "spectral_null_commutator": [],
            }

    for split_seed in args.split_seeds:
        for fold, (train_indices, test_indices) in enumerate(fold_indices(len(target), args.folds, split_seed)):
            x_train_raw, x_test_raw = target[train_indices], target[test_indices]
            y_train_raw, y_test_raw = predicted[train_indices], predicted[test_indices]
            x_mean, y_mean = x_train_raw.mean(0), y_train_raw.mean(0)
            x_train = x_train_raw - x_mean
            x_test = x_test_raw - x_mean
            y_train = y_train_raw - y_mean
            y_test = y_test_raw - y_mean
            _, _, vh = torch.linalg.svd(x_train, full_matrices=False)

            for rank in args.ranks:
                effective_rank = min(rank, len(train_indices) - 1, vh.shape[0])
                if effective_rank != rank:
                    continue
                basis = vh[:rank]
                x_train_reduced = x_train @ basis.T
                x_test_reduced = x_test @ basis.T
                y_train_reduced = y_train @ basis.T
                y_test_reduced = y_test @ basis.T

                for ridge in args.ridge:
                    key = f"rank={rank},ridge={ridge:g}"
                    identity = torch.eye(rank, dtype=x_test_reduced.dtype)
                    records[key]["identity_r2"].append(score(x_test_reduced, y_test_reduced, identity)[0])
                    records[key]["zero_r2"].append(score(x_test_reduced, y_test_reduced, torch.zeros_like(identity))[0])
                    paired = fit_ridge(x_train_reduced, y_train_reduced, ridge)
                    r2, rel_mse, comm = score(x_test_reduced, y_test_reduced, paired)
                    records[key]["paired_r2"].append(r2)
                    records[key]["paired_relative_mse"].append(rel_mse)
                    records[key]["paired_commutator"].append(comm)

                    for control_seed in args.control_seeds:
                        generator = torch.Generator().manual_seed(
                            10_000 * split_seed + 1_000 * fold + control_seed
                        )
                        shuffled_y = y_train_reduced[torch.randperm(len(y_train_reduced), generator=generator)]
                        shuffled = fit_ridge(x_train_reduced, shuffled_y, ridge)
                        r2, rel_mse, comm = score(x_test_reduced, y_test_reduced, shuffled)
                        records[key]["shuffled_r2"].append(r2)
                        records[key]["shuffled_relative_mse"].append(rel_mse)
                        records[key]["shuffled_commutator"].append(comm)

                        q = orthogonal(rank, 1_000_000 * split_seed + 10_000 * fold + control_seed)
                        spectral_null = q @ paired @ q.T
                        r2, rel_mse, comm = score(x_test_reduced, y_test_reduced, spectral_null)
                        records[key]["spectral_null_r2"].append(r2)
                        records[key]["spectral_null_relative_mse"].append(rel_mse)
                        records[key]["spectral_null_commutator"].append(comm)

    sweep = {
        key: {metric: summarize(values) for metric, values in metrics.items()}
        for key, metrics in records.items()
        if metrics["paired_r2"]
    }
    primary_key = f"rank={args.primary_rank},ridge={args.primary_ridge:g}"
    if primary_key not in sweep:
        raise ValueError(f"Primary setting {primary_key} is absent from the requested sweep")
    primary = sweep[primary_key]
    primary["paired_minus_shuffled_r2"] = {
        "mean_difference": primary["paired_r2"]["mean"] - primary["shuffled_r2"]["mean"]
    }
    primary["paired_minus_spectral_null_r2"] = {
        "mean_difference": primary["paired_r2"]["mean"] - primary["spectral_null_r2"]["mean"]
    }
    return {"primary": primary, "sweep": sweep}


def main() -> None:
    args = parse_args()
    if args.label and len(args.label) != len(args.input):
        raise ValueError("Provide exactly one --label for each --input")
    labels = args.label or [path.stem for path in args.input]
    output = {
        "protocol": {
            "folds": args.folds,
            "split_seeds": args.split_seeds,
            "control_seeds": args.control_seeds,
            "ranks": args.ranks,
            "ridge": args.ridge,
            "primary_setting": {"rank": args.primary_rank, "ridge": args.primary_ridge},
            "fit_direction": "target latent -> predicted latent",
            "leakage_control": "All means, PCA bases, ridge scales, and operators are fit on training folds only.",
            "ridge_scaling": "lambda times mean diagonal of the training-fold Gram matrix",
        },
        "models": {},
        "evidence_boundary": (
            "Cross-fitted identification and estimator-sensitivity analysis for the supplied released-checkpoint "
            "replays. It tests whether paired lower-rank linear structure survives held-out episodes; it does not "
            "rank model families or prove that arbitrary learned world models obey the finite-dimensional theorem."
        ),
    }
    for label, path in zip(labels, args.input):
        payload = json.loads(path.read_text())
        vectors = payload.get("operator_fit_vectors")
        if not vectors:
            raise ValueError(f"{path} does not contain operator_fit_vectors; rerun evaluator with --save-vectors")
        model_result = {
            "source": str(path),
            "checkpoint_sha256": payload.get("protocol", {}).get("checkpoint_sha256"),
            "holdout_episode_count": payload.get("protocol", {}).get("holdout_episode_count"),
            "horizons": {},
        }
        for horizon in sorted(vectors["target"], key=int):
            target = torch.tensor(vectors["target"][horizon], dtype=torch.float64)
            predicted = torch.tensor(vectors["predicted"][horizon], dtype=torch.float64)
            if target.shape != predicted.shape or target.ndim != 2:
                raise ValueError(f"Invalid vectors for {label}, horizon {horizon}: {target.shape}, {predicted.shape}")
            model_result["horizons"][horizon] = analyze_horizon(target, predicted, args)
        output["models"][label] = model_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    compact = {
        label: {
            horizon: {
                "paired_r2": values["primary"]["paired_r2"]["mean"],
                "shuffled_r2": values["primary"]["shuffled_r2"]["mean"],
                "spectral_null_r2": values["primary"]["spectral_null_r2"]["mean"],
                "identity_r2": values["primary"]["identity_r2"]["mean"],
                "paired_commutator": values["primary"]["paired_commutator"]["mean"],
            }
            for horizon, values in result["horizons"].items()
        }
        for label, result in output["models"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
