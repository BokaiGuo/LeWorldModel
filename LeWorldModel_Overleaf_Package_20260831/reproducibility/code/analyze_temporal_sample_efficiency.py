"""Training-episode ablation for cross-fitted temporal latent operators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from analyze_temporal_operator_controls import fit_ridge, orthogonal, score, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[20, 40, 80, 120, 160])
    parser.add_argument("--test-size", type=int, default=40)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--control-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    return parser.parse_args()


def evaluate(
    target: torch.Tensor,
    predicted: torch.Tensor,
    train_size: int,
    args: argparse.Namespace,
) -> dict:
    values = {"paired_r2": [], "shuffled_r2": [], "spectral_null_r2": [], "paired_commutator": []}
    n = len(target)
    if train_size + args.test_size > n:
        raise ValueError(f"Need {train_size + args.test_size} samples but only {n} are available")
    for split_seed in args.split_seeds:
        permutation = torch.randperm(n, generator=torch.Generator().manual_seed(split_seed))
        test_indices = permutation[-args.test_size:]
        train_indices = permutation[:train_size]
        x_train_raw, x_test_raw = target[train_indices], target[test_indices]
        y_train_raw, y_test_raw = predicted[train_indices], predicted[test_indices]
        x_mean, y_mean = x_train_raw.mean(0), y_train_raw.mean(0)
        x_train, x_test = x_train_raw - x_mean, x_test_raw - x_mean
        y_train, y_test = y_train_raw - y_mean, y_test_raw - y_mean
        _, _, vh = torch.linalg.svd(x_train, full_matrices=False)
        basis = vh[: args.rank]
        x_train, x_test = x_train @ basis.T, x_test @ basis.T
        y_train, y_test = y_train @ basis.T, y_test @ basis.T
        operator = fit_ridge(x_train, y_train, args.ridge)
        r2, _, commutator = score(x_test, y_test, operator)
        values["paired_r2"].append(r2)
        values["paired_commutator"].append(commutator)
        for control_seed in args.control_seeds:
            generator = torch.Generator().manual_seed(10_000 * split_seed + control_seed)
            shuffled = fit_ridge(
                x_train,
                y_train[torch.randperm(len(y_train), generator=generator)],
                args.ridge,
            )
            values["shuffled_r2"].append(score(x_test, y_test, shuffled)[0])
            q = orthogonal(args.rank, 1_000_000 * split_seed + control_seed)
            values["spectral_null_r2"].append(score(x_test, y_test, q @ operator @ q.T)[0])
    return {key: summarize(items) for key, items in values.items()}


def main() -> None:
    args = parse_args()
    if len(args.input) != len(args.label):
        raise ValueError("Provide one --label per --input")
    output = {
        "protocol": {
            "train_sizes": args.train_sizes,
            "test_size": args.test_size,
            "rank": args.rank,
            "ridge": args.ridge,
            "split_seeds": args.split_seeds,
            "control_seeds": args.control_seeds,
            "nested_training_subsets": True,
            "test_set": "fixed final test_size entries of each seeded permutation",
        },
        "models": {},
        "evidence_boundary": "Sample-efficiency ablation on stored checkpoint latents; no model training or task-level claim.",
    }
    for path, label in zip(args.input, args.label):
        payload = json.loads(path.read_text())
        vectors = payload["operator_fit_vectors"]
        result = {"source": str(path), "horizons": {}}
        for horizon in sorted(vectors["target"], key=int):
            target = torch.tensor(vectors["target"][horizon], dtype=torch.float64)
            predicted = torch.tensor(vectors["predicted"][horizon], dtype=torch.float64)
            result["horizons"][horizon] = {
                str(size): evaluate(target, predicted, size, args) for size in args.train_sizes
            }
        output["models"][label] = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    compact = {
        model: {
            horizon: {size: round(scores["paired_r2"]["mean"], 4) for size, scores in sizes.items()}
            for horizon, sizes in result["horizons"].items()
        }
        for model, result in output["models"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
