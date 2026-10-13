"""Paired statistical tests for action-intervention checkpoint replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed", type=Path, action="append", required=True)
    parser.add_argument("--zero", type=Path, action="append", required=True)
    parser.add_argument("--reverse", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-trials", type=int, default=20000)
    parser.add_argument("--permutation-trials", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if "operator_fit_vectors" not in payload:
        raise ValueError(f"{path} lacks operator_fit_vectors")
    return payload


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values in input order."""
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def paired_stats(
    delta: torch.Tensor,
    bootstrap_trials: int,
    permutation_trials: int,
    seed: int,
    exact_threshold: int = 21,
) -> dict:
    delta = delta.to(torch.float64)
    n = len(delta)
    generator = torch.Generator().manual_seed(seed)
    bootstrap_means = []
    chunk = 1000
    for start in range(0, bootstrap_trials, chunk):
        count = min(chunk, bootstrap_trials - start)
        indices = torch.randint(0, n, (count, n), generator=generator)
        bootstrap_means.append(delta[indices].mean(1))
    bootstrap_means = torch.cat(bootstrap_means)

    observed = delta.mean().abs()
    extreme = 0
    completed = 0
    if n <= exact_threshold:
        total = 1 << n
        sign_chunk = 100_000
        shifts = torch.arange(n, dtype=torch.int64)
        for start in range(0, total, sign_chunk):
            count = min(sign_chunk, total - start)
            states = torch.arange(start, start + count, dtype=torch.int64)
            signs = ((states[:, None] >> shifts) & 1).to(torch.float64).mul_(2).sub_(1)
            permuted = (signs * delta).mean(1).abs()
            extreme += int(torch.sum(permuted >= observed))
            completed += count
        p_value = extreme / completed
        method = "exact_enumeration"
    else:
        for start in range(0, permutation_trials, chunk):
            count = min(chunk, permutation_trials - start)
            signs = torch.randint(0, 2, (count, n), generator=generator, dtype=torch.int64)
            signs = signs.to(torch.float64).mul_(2).sub_(1)
            permuted = (signs * delta).mean(1).abs()
            extreme += int(torch.sum(permuted >= observed))
            completed += count
        p_value = (extreme + 1) / (completed + 1)
        method = "monte_carlo"

    std = delta.std(unbiased=True)
    return {
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
        "bootstrap_ci95": [
            float(torch.quantile(bootstrap_means, 0.025)),
            float(torch.quantile(bootstrap_means, 0.975)),
        ],
        "sign_flip_p_two_sided": p_value,
        "sign_flip_method": method,
        "sign_flip_permutations_evaluated": completed,
        "paired_effect_size_dz": float(delta.mean() / std) if float(std) > 0 else None,
        "fraction_intervention_worse": float((delta > 0).to(torch.float64).mean()),
        "episodes": n,
    }


def mse_vectors(payload: dict, horizon: str) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = payload["operator_fit_vectors"]
    predicted = torch.tensor(vectors["predicted"][horizon], dtype=torch.float64)
    target = torch.tensor(vectors["target"][horizon], dtype=torch.float64)
    return ((predicted - target) ** 2).mean(1), target


def main() -> None:
    args = parse_args()
    count = len(args.label)
    if not (len(args.observed) == len(args.zero) == len(args.reverse) == count):
        raise ValueError("Provide one observed/zero/reverse triplet per label")
    output = {
        "protocol": {
            "bootstrap_trials": args.bootstrap_trials,
            "permutation_trials": args.permutation_trials,
            "seed": args.seed,
            "unit": "episode",
            "test": "paired bootstrap CI plus two-sided sign-flip permutation test",
            "exact_sign_flip_max_episodes": 21,
            "multiple_testing": "Holm family-wise-error correction across all model, horizon, and intervention cells in this artifact",
        },
        "models": {},
        "evidence_boundary": "Paired checkpoint-replay sensitivity, not causal task performance or model ranking.",
    }
    for model_index, (label, observed_path, zero_path, reverse_path) in enumerate(
        zip(args.label, args.observed, args.zero, args.reverse)
    ):
        observed, zero, reverse = load(observed_path), load(zero_path), load(reverse_path)
        model = {
            "sources": {"observed": str(observed_path), "zero": str(zero_path), "reverse": str(reverse_path)},
            "target_vectors_identical": True,
            "episode_order_identical": (
                observed.get("protocol", {}).get("episodes_used")
                == zero.get("protocol", {}).get("episodes_used")
                == reverse.get("protocol", {}).get("episodes_used")
            ),
            "horizons": {},
        }
        horizons = sorted(observed["operator_fit_vectors"]["target"], key=int)
        for horizon in horizons:
            observed_mse, target_observed = mse_vectors(observed, horizon)
            zero_mse, target_zero = mse_vectors(zero, horizon)
            reverse_mse, target_reverse = mse_vectors(reverse, horizon)
            targets_identical = torch.equal(target_observed, target_zero) and torch.equal(target_observed, target_reverse)
            model["target_vectors_identical"] = model["target_vectors_identical"] and targets_identical
            if not targets_identical:
                raise ValueError(f"Target vectors differ for {label}, horizon {horizon}")
            model["horizons"][horizon] = {
                "observed_mse_mean": float(observed_mse.mean()),
                "zero_mse_mean": float(zero_mse.mean()),
                "reverse_mse_mean": float(reverse_mse.mean()),
                "zero_minus_observed": paired_stats(
                    zero_mse - observed_mse,
                    args.bootstrap_trials,
                    args.permutation_trials,
                    args.seed + 1000 * model_index + 10 * int(horizon),
                ),
                "reverse_minus_observed": paired_stats(
                    reverse_mse - observed_mse,
                    args.bootstrap_trials,
                    args.permutation_trials,
                    args.seed + 1000 * model_index + 10 * int(horizon) + 1,
                ),
            }
        output["models"][label] = model
    tests = []
    for model in output["models"].values():
        for horizon in model["horizons"].values():
            tests.extend([horizon["zero_minus_observed"], horizon["reverse_minus_observed"]])
    for result, adjusted in zip(tests, holm_adjust([row["sign_flip_p_two_sided"] for row in tests])):
        result["holm_adjusted_p"] = adjusted
        result["holm_reject_0_05"] = adjusted < 0.05

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    compact = {
        label: {
            horizon: {
                mode: {
                    "delta": round(values[mode]["mean_delta"], 6),
                    "ci95": [round(x, 6) for x in values[mode]["bootstrap_ci95"]],
                    "p": values[mode]["sign_flip_p_two_sided"],
                }
                for mode in ["zero_minus_observed", "reverse_minus_observed"]
            }
            for horizon, values in model["horizons"].items()
        }
        for label, model in output["models"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
