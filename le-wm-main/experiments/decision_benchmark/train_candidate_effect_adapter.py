"""Train and evaluate an intervention-supervised candidate-effect adapter.

This is an explicit causal-data intervention, not a frozen-geometry result.
The adapter is trained only on train-root simulator outcomes from one shared
equal-duration candidate bank.  It receives the frozen RGB-derived source
latent and the complete action sequence, then predicts terminal XY effect.
Outcome roots are never used for fitting, normalization, or model selection.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


SEEDS = (8411, 8412, 8413, 8414, 8415)


class EffectMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 2),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _array_key(reference: str) -> str:
    parts = str(reference).split("::", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid array reference: {reference}")
    return parts[1]


def _features(
    records: list[dict[str, Any]],
    arrays: Any,
    feature_mode: str = "frozen_latent_action",
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    if feature_mode not in {
        "frozen_latent_action",
        "history_latent_action",
        "privileged_state_action",
        "latent_plus_state_action",
    }:
        raise ValueError(f"unsupported feature_mode: {feature_mode}")
    x_values, y_values, roots, goals = [], [], [], []
    for row in records:
        if row.get("source_latent_model") is None:
            raise ValueError("candidate record lacks source_latent_model; regenerate the bank")
        source = np.asarray(row["source_latent_model"], dtype=np.float64)
        action = np.asarray(row["action_sequence"], dtype=np.float64).reshape(-1)
        realized = np.asarray(arrays[_array_key(row["realized_terminal_state_ref"])], dtype=np.float64)
        source_state = np.asarray(row["source_state_raw"], dtype=np.float64)
        if feature_mode == "frozen_latent_action":
            context = source
        elif feature_mode == "history_latent_action":
            if row.get("source_history_model") is None:
                raise ValueError("candidate record lacks source_history_model; regenerate the bank")
            context = np.asarray(row["source_history_model"], dtype=np.float64)
        elif feature_mode == "privileged_state_action":
            context = source_state
        else:
            context = np.concatenate([source, source_state])
        x_values.append(np.concatenate([context, action]))
        y_values.append(realized[:2] - source_state[:2])
        roots.append(str(row["instance_id"]))
        goals.append(np.asarray(row["goal_raw"], dtype=np.float64))
    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32), roots, np.asarray(goals, dtype=np.float64)


def _resolve_device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported device: {requested}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    seed: int,
    epochs: int,
    device_name: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    _seed(seed)
    device = _resolve_device(device_name)
    x_mean = train_x.mean(axis=0)
    x_scale = np.maximum(train_x.std(axis=0), 1e-3)
    y_mean = train_y.mean(axis=0)
    y_scale = np.maximum(train_y.std(axis=0), 1e-3)
    model = EffectMLP(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    x = torch.as_tensor((train_x - x_mean) / x_scale, dtype=torch.float32, device=device)
    y = torch.as_tensor((train_y - y_mean) / y_scale, dtype=torch.float32, device=device)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        pred = model(torch.as_tensor(
            (eval_x - x_mean) / x_scale, dtype=torch.float32, device=device,
        )).cpu().numpy()
    pred = pred * y_scale + y_mean
    return pred, {
        "x_mean": x_mean.tolist(), "x_scale": x_scale.tolist(),
        "y_mean": y_mean.tolist(), "y_scale": y_scale.tolist(),
        "epochs": int(epochs),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "device": str(device),
    }


def _bootstrap(values: np.ndarray, roots: list[str], seed: int, samples: int = 1000) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    unique = np.asarray(sorted(set(roots)))
    if len(unique) < 2:
        mean = float(np.mean(values))
        return {"estimate": mean, "lower": mean, "upper": mean, "groups": int(len(unique))}
    rng = np.random.default_rng(seed)
    grouped = [values[np.asarray(roots) == root] for root in unique]
    draws = []
    for _ in range(samples):
        indices = rng.integers(0, len(unique), len(unique))
        draws.append(float(np.mean(np.concatenate([grouped[index] for index in indices]))))
    return {"estimate": float(values.mean()), "lower": float(np.quantile(draws, .025)), "upper": float(np.quantile(draws, .975)), "groups": int(len(unique))}


def run(
    dataset: Path,
    arrays: Path,
    output: Path,
    *,
    epochs: int = 300,
    bootstrap_samples: int = 1000,
    feature_mode: str = "frozen_latent_action",
    device: str = "auto",
) -> dict[str, Any]:
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    all_records = payload["records"]
    train_records = [row for row in all_records if row.get("split") == "train"]
    outcome_records = [row for row in all_records if row.get("split") == "outcome"]
    if not train_records or not outcome_records:
        raise ValueError("train and outcome records are required")
    arrays_payload = np.load(arrays, allow_pickle=False)
    train_x, train_y, train_roots, _ = _features(train_records, arrays_payload, feature_mode)
    outcome_x, outcome_y, outcome_roots, outcome_goals = _features(outcome_records, arrays_payload, feature_mode)
    outcome_sources = np.asarray([row["source_state_raw"] for row in outcome_records], dtype=np.float64)[:, :2]
    grouped_outcome: dict[str, list[int]] = {}
    for index, root in enumerate(outcome_roots):
        grouped_outcome.setdefault(root, []).append(index)
    reports = []
    for seed in SEEDS:
        prediction, training = _fit_predict(train_x, train_y, outcome_x, seed, epochs, device)
        train_prediction, _ = _fit_predict(train_x, train_y, train_x, seed, epochs, device)
        predicted_terminal = outcome_sources + prediction
        predicted_cost = np.linalg.norm(predicted_terminal - outcome_goals, axis=1)
        actual_cost = np.linalg.norm(outcome_sources + outcome_y - outcome_goals, axis=1)
        selected_errors, random_errors, oracle_errors = [], [], []
        selected_success, random_success = [], []
        for root, indices in sorted(grouped_outcome.items()):
            indices = np.asarray(indices, dtype=np.int64)
            selected = indices[int(np.argmin(predicted_cost[indices]))]
            random_index = indices[int(np.random.default_rng(seed + len(root)).integers(0, len(indices)))]
            oracle = float(np.min(actual_cost[indices]))
            selected_errors.append(float(actual_cost[selected] - oracle))
            random_errors.append(float(actual_cost[random_index] - oracle))
            oracle_errors.append(0.0)
            selected_success.append(float(actual_cost[selected] <= .25))
            random_success.append(float(actual_cost[random_index] <= .25))
        paired_regret_gain = np.asarray(random_errors, dtype=np.float64) - np.asarray(selected_errors, dtype=np.float64)
        paired_success_gain = np.asarray(selected_success, dtype=np.float64) - np.asarray(random_success, dtype=np.float64)
        reports.append({
            "seed": int(seed),
            "effect_rmse": float(np.sqrt(np.mean((prediction - outcome_y) ** 2))),
            "train_effect_rmse": float(np.sqrt(np.mean((train_prediction - train_y) ** 2))),
            "selected_regret": _bootstrap(np.asarray(selected_errors), sorted(grouped_outcome), seed + 100, bootstrap_samples),
            "random_regret": _bootstrap(np.asarray(random_errors), sorted(grouped_outcome), seed + 200, bootstrap_samples),
            "selected_success": _bootstrap(np.asarray(selected_success), sorted(grouped_outcome), seed + 300, bootstrap_samples),
            "random_success": _bootstrap(np.asarray(random_success), sorted(grouped_outcome), seed + 400, bootstrap_samples),
            "paired_regret_gain": _bootstrap(paired_regret_gain, sorted(grouped_outcome), seed + 500, bootstrap_samples),
            "paired_success_gain": _bootstrap(paired_success_gain, sorted(grouped_outcome), seed + 600, bootstrap_samples),
            "training": training,
        })
    report = {
        "schema": "pointmaze_intervention_supervised_candidate_effect.v1",
        "dataset": str(dataset.resolve()),
        "arrays": str(arrays.resolve()),
        "reports": reports,
        "summary": {
            "selected_regret_mean": float(np.mean([item["selected_regret"]["estimate"] for item in reports])),
            "random_regret_mean": float(np.mean([item["random_regret"]["estimate"] for item in reports])),
            "effect_rmse_mean": float(np.mean([item["effect_rmse"] for item in reports])),
            "train_effect_rmse_mean": float(np.mean([item["train_effect_rmse"] for item in reports])),
            "paired_regret_gain_mean": float(np.mean([item["paired_regret_gain"]["estimate"] for item in reports])),
            "paired_success_gain_mean": float(np.mean([item["paired_success_gain"]["estimate"] for item in reports])),
        },
        "protocol": {
            "training_split": "train roots only",
            "outcome_split": "root-disjoint outcome roots",
            "labels": "offline simulator terminal XY effect",
            "runtime_input": {
                "frozen_latent_action": "frozen source latent plus complete action sequence",
                "history_latent_action": "causal frozen latent history up to the decision point plus complete action sequence",
                "privileged_state_action": "privileged source simulator state plus complete action sequence (diagnostic upper bound only)",
                "latent_plus_state_action": "frozen source latent plus privileged source simulator state plus complete action sequence (diagnostic upper bound only)",
            }[feature_mode],
            "privileged_state_at_runtime": feature_mode in {"privileged_state_action", "latent_plus_state_action"},
            "outcome_used_for_training_or_selection": False,
            "adapter_role": "intervention_supervised",
            "feature_mode": feature_mode,
            "device": str(device),
            "feature_mode_boundary": "privileged_state modes are diagnostic upper bounds and not RGB-runtime evidence",
        },
        "claim_boundary": "Intervention-supervised candidate-effect adapter; not frozen geometry evidence, not online RGB planning, and not universal transfer.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--feature-mode",
        choices=("frozen_latent_action", "history_latent_action", "privileged_state_action", "latent_plus_state_action"),
        default="frozen_latent_action",
    )
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
