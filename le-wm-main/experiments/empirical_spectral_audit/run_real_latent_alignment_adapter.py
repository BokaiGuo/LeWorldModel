"""Post-hoc alignment intervention on a frozen LeWM PushT latent artifact.

This experiment does not retrain the pixel encoder or the LeWM predictor.  It
fits a small residual adapter on the already exported frozen latent pairs:

    z_next_hat = z + predicted_dynamics + correction(z, predicted_dynamics, action).

The correction is trained against the exported one-step dynamics.  The same
row-level split, initialization, optimizer, and evaluation are used for
``none``, ``global``, and ``conditional`` alignment.  Because the compact
artifact does not contain episode IDs/timestamps, this script intentionally
reports held-out one-step prediction only; it does not manufacture rollout or
planning evidence.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from experiments.empirical_spectral_audit.regularizers import (
        operator_alignment_diagnostic,
        operator_alignment_loss,
    )
except ModuleNotFoundError:  # direct invocation
    from regularizers import operator_alignment_diagnostic, operator_alignment_loss


@dataclass(frozen=True)
class AdapterConfig:
    epochs: int = 20
    batch_size: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    alignment_bins: int = 2
    alignment_min_samples: int = 64
    alignment_ridge: float = 1e-3


class ResidualAdapter(torch.nn.Module):
    """Zero-initialized linear correction over frozen latent transition features."""

    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super().__init__()
        self.correction = torch.nn.Linear(2 * latent_dim + action_dim, latent_dim)
        torch.nn.init.zeros_(self.correction.weight)
        torch.nn.init.zeros_(self.correction.bias)

    def forward(self, source: torch.Tensor, base_delta: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = torch.cat((source, base_delta, action), dim=-1)
        return source + base_delta + self.correction(features)


def _load_arrays(path: Path, max_samples: int, split_seed: int) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = {"latents", "dynamics", "predicted_dynamics", "actions"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"missing required latent fields: {sorted(missing)}")
        n = int(data["latents"].shape[0])
        rng = np.random.default_rng(split_seed)
        rows = np.arange(n)
        if max_samples > 0 and n > max_samples:
            rows = np.sort(rng.choice(n, size=max_samples, replace=False))
        arrays = {name: np.asarray(data[name][rows], dtype=np.float32) for name in required}
    return arrays


def _split_indices(n: int, val_fraction: float, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    test = order[:n_test]
    val = order[n_test : n_test + n_val]
    train = order[n_test + n_val :]
    return train, val, test


def _tensor_dataset(arrays: dict[str, np.ndarray], indices: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(arrays["latents"][indices]),
        torch.from_numpy(arrays["predicted_dynamics"][indices]),
        torch.from_numpy(arrays["dynamics"][indices]),
        torch.from_numpy(arrays["actions"][indices]),
    )


@torch.no_grad()
def _evaluate(
    model: ResidualAdapter,
    loader: DataLoader,
    device: torch.device,
    alignment_bins: int,
    alignment_min_samples: int,
    alignment_ridge: float,
) -> dict[str, object]:
    errors = []
    sources = []
    predictions = []
    actions = []
    for source, base_delta, target_delta, action in loader:
        source = source.to(device)
        base_delta = base_delta.to(device)
        target_delta = target_delta.to(device)
        action = action.to(device)
        prediction = model(source, base_delta, action)
        errors.append((prediction - (source + target_delta)).detach().cpu())
        sources.append(source.detach().cpu())
        predictions.append(prediction.detach().cpu())
        actions.append(action.detach().cpu())
    error = torch.cat(errors)
    source = torch.cat(sources)
    prediction = torch.cat(predictions)
    action = torch.cat(actions)
    labels = torch.bucketize(
        action.norm(dim=-1),
        torch.quantile(action.norm(dim=-1), torch.tensor([0.5])),
    )
    mse_by_bin = {
        str(group): float(error[labels == group].square().mean())
        for group in (0, 1)
        if bool((labels == group).any())
    }
    global_alignment = operator_alignment_diagnostic(
        source, prediction, mode="global", min_samples=alignment_min_samples, ridge=alignment_ridge
    )
    conditional_alignment = operator_alignment_diagnostic(
        source,
        prediction,
        actions=action,
        mode="conditional",
        num_bins=alignment_bins,
        min_samples=alignment_min_samples,
        ridge=alignment_ridge,
    )
    return {
        "mse": float(error.square().mean()),
        "mse_by_action_norm_bin": mse_by_bin,
        "global_alignment": global_alignment,
        "conditional_alignment": conditional_alignment,
        "error_rmse": float(error.square().mean().sqrt()),
    }


def _frozen_baseline(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, object]:
    error = arrays["predicted_dynamics"][indices] - arrays["dynamics"][indices]
    source = torch.from_numpy(arrays["latents"][indices])
    prediction = source + torch.from_numpy(arrays["predicted_dynamics"][indices])
    actions = torch.from_numpy(arrays["actions"][indices])
    return {
        "mse": float(np.mean(error * error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "global_alignment": operator_alignment_diagnostic(source, prediction, mode="global"),
        "conditional_alignment": operator_alignment_diagnostic(
            source, prediction, actions=actions, mode="conditional", num_bins=2
        ),
    }


def _run_one(
    arrays: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    mode: str,
    weight: float,
    seed: int,
    config: AdapterConfig,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    model = ResidualAdapter(arrays["latents"].shape[1], arrays["actions"].shape[1]).to(device)
    train_loader = DataLoader(_tensor_dataset(arrays, train_idx), batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(_tensor_dataset(arrays, val_idx), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(_tensor_dataset(arrays, test_idx), batch_size=config.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_val = float("inf")
    best_state = None
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_total = 0.0
        pred_total = 0.0
        align_total = 0.0
        count = 0
        for source, base_delta, target_delta, action in train_loader:
            source = source.to(device)
            base_delta = base_delta.to(device)
            target_delta = target_delta.to(device)
            action = action.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(source, base_delta, action)
            prediction_loss = (prediction - (source + target_delta)).square().mean()
            alignment_loss = operator_alignment_loss(
                source,
                prediction,
                actions=action,
                mode=mode,
                num_bins=config.alignment_bins,
                min_samples=config.alignment_min_samples,
                ridge=config.alignment_ridge,
            )
            loss = prediction_loss + float(weight) * alignment_loss
            loss.backward()
            optimizer.step()
            batch_n = source.size(0)
            count += batch_n
            loss_total += float(loss.detach().cpu()) * batch_n
            pred_total += float(prediction_loss.detach().cpu()) * batch_n
            align_total += float(alignment_loss.detach().cpu()) * batch_n
        validation = _evaluate(
            model, val_loader, device, config.alignment_bins, config.alignment_min_samples, config.alignment_ridge
        )
        row = {
            "epoch": epoch,
            "train_objective": loss_total / count,
            "train_prediction_mse": pred_total / count,
            "train_alignment_loss": align_total / count,
            "val_mse": validation["mse"],
        }
        history.append(row)
        if float(validation["mse"]) < best_val:
            best_val = float(validation["mse"])
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no best adapter state was recorded")
    model.load_state_dict(best_state)
    test = _evaluate(
        model, test_loader, device, config.alignment_bins, config.alignment_min_samples, config.alignment_ridge
    )
    return {
        "mode": mode,
        "alignment_weight": weight,
        "seed": seed,
        "best_val_mse": best_val,
        "test": test,
        "history": history,
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_test": int(len(test_idx)),
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(f"{record['mode']}:{record['alignment_weight']}", []).append(record)
    baseline_by_seed = {
        int(row["seed"]): float(row["test"]["mse"])
        for row in grouped.get("none:0.0", [])
    }
    out = {}
    for key, rows in grouped.items():
        values = [float(row["test"]["mse"]) for row in rows]  # type: ignore[index]
        align = [float(row["test"]["conditional_alignment"]["operator_alignment"]) for row in rows]  # type: ignore[index]
        paired_delta = [
            float(row["test"]["mse"]) - baseline_by_seed[int(row["seed"])]
            for row in rows
            if int(row["seed"]) in baseline_by_seed
        ]
        out[key] = {
            "n_seeds": len(rows),
            "test_mse_mean": _mean(values),
            "test_mse_std": float(np.std(values)),
            "conditional_alignment_mean": _mean(align),
            "test_mse_values": values,
            "paired_mse_delta_vs_none_mean": _mean(paired_delta) if paired_delta else None,
            "paired_mse_delta_vs_none_std": float(np.std(paired_delta)) if paired_delta else None,
            "paired_mse_relative_delta_vs_none_mean": (
                _mean([delta / baseline_by_seed[int(row["seed"])] for delta, row in zip(paired_delta, rows)])
                if paired_delta
                else None
            ),
        }
    return out


def _write_markdown(report: dict[str, object], path: Path) -> None:
    protocol = report["protocol"]
    lines = [
        "# Frozen PushT LeWM latent alignment adapter",
        "",
        "This is a post-hoc one-step latent adapter on a frozen LeWM checkpoint artifact. It is not end-to-end LeWM retraining and does not provide rollout or online-planning evidence.",
        "",
        "## Protocol",
        "",
        f"- source artifact: `{protocol['latents']}`; rows: `{protocol['rows']}`; latent dimension: `{protocol['latent_dim']}`",
        f"- split: row-level train/validation/test = `{protocol['split']}`; seeds: `{protocol['seeds']}`; epochs: `{protocol['epochs']}`",
        "- target: exported `dynamics`; frozen baseline input: exported `predicted_dynamics`; adapter receives latent, frozen displacement, and action",
        "- no episode IDs/timestamps are present, so rollout/planning metrics are intentionally absent",
        "",
        "## Results",
        "",
        f"Frozen no-adapter baseline: held-out one-step MSE `{report['frozen_baseline_test']['mse']:.6g}`; conditional alignment `{report['frozen_baseline_test']['conditional_alignment']['operator_alignment']:.6g}`.",
        "The learned residual adapter without alignment reduces one-step MSE, so alignment-specific effects must be judged relative to the `none` adapter row, not relative to the frozen predictor alone.",
        "",
        "| mode | weight | seeds | held-out one-step MSE | std | paired delta vs none | conditional alignment |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in report["summary"].items():
        mode, weight = key.split(":", 1)
        lines.append(
            f"| {mode} | {float(weight):.6g} | {values['n_seeds']} | {values['test_mse_mean']:.6g} | "
            f"{values['test_mse_std']:.3g} | {values['paired_mse_delta_vs_none_mean']:.3g} | "
            f"{values['conditional_alignment_mean']:.6g} |"
        )
    lines.extend([
        "",
        "## Claim boundary",
        "",
        "This artifact tests whether an alignment-constrained correction can alter held-out one-step latent prediction on real PushT samples. It cannot establish end-to-end training gains, multi-step rollout gains, online control gains, or planning success because the compact latent export lacks episode/timestep structure and the original HDF5 is not available in expanded form.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.001, 0.01, 0.1])
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    config = AdapterConfig(epochs=args.epochs, batch_size=args.batch_size)
    arrays = _load_arrays(args.latents, args.max_samples, split_seed=3072)
    train_idx, val_idx, test_idx = _split_indices(
        len(arrays["latents"]), config.val_fraction, config.test_fraction, seed=3072
    )
    frozen = _frozen_baseline(arrays, test_idx)
    records = []
    start = time.time()
    for mode in ("none", "global", "conditional"):
        weights = [0.0] if mode == "none" else args.weights
        for weight in weights:
            for seed in args.seeds:
                print(f"[real-adapter] mode={mode} weight={weight} seed={seed}: measuring held-out one-step consequence", flush=True)
                records.append(_run_one(arrays, train_idx, val_idx, test_idx, mode, weight, seed, config, device))
    report = {
        "protocol": {
            "latents": str(args.latents),
            "rows": int(len(arrays["latents"])),
            "latent_dim": int(arrays["latents"].shape[1]),
            "action_dim": int(arrays["actions"].shape[1]),
            "split": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
            "seeds": args.seeds,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "weights": args.weights,
            "device": str(device),
            "frozen_checkpoint": True,
            "end_to_end_retraining": False,
            "rollout_and_planning_available": False,
            "elapsed_seconds": time.time() - start,
        },
        "frozen_baseline_test": frozen,
        "records": records,
        "summary": _summary(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
