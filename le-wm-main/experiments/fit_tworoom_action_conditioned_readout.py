"""Fit an episode-disjoint action-conditioned physical successor readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _fit(x: np.ndarray, y: np.ndarray, ridge: float):
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    xs = (x - mean) / scale
    design = np.column_stack([xs, np.ones(len(xs))])
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return coef, mean, scale


def _predict(model, x):
    coef, mean, scale = model
    return np.column_stack([(x - mean) / scale, np.ones(len(x))]) @ coef


def _metrics(model, x, y):
    error = _predict(model, x) - y
    return {
        "n": int(len(x)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "target_std": float(np.mean(np.std(y, axis=0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--ridge", type=float, default=1e-2)
    args = parser.parse_args()
    pred_key = f"predicted_dynamics_h{args.horizon}"
    with np.load(args.latents) as data:
        required = {"latents", pred_key, "actions", "physical_proprio", "episode_id", "timestep"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"missing arrays: {missing}")
        latents = np.asarray(data["latents"], dtype=np.float64)
        predicted_terminal = latents + np.asarray(data[pred_key], dtype=np.float64)
        actions = np.asarray(data["actions"], dtype=np.float64)[:, :10]
        physical = np.asarray(data["physical_proprio"], dtype=np.float64).reshape(len(latents), -1)[:, :2]
        episode = np.asarray(data["episode_id"]).reshape(-1).astype(np.int64)
        timestep = np.asarray(data["timestep"]).reshape(-1).astype(np.int64)
    order = np.lexsort((timestep, episode))
    keys = np.stack([episode[order], timestep[order]], axis=1)
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = np.any(keys[1:] != keys[:-1], axis=1)
    chosen = order[keep]
    current_map = {(int(episode[i]), int(timestep[i])): physical[i] for i in chosen}
    valid: list[int] = []
    targets: list[np.ndarray] = []
    for index in chosen:
        target = current_map.get((int(episode[index]), int(timestep[index]) + args.horizon))
        if target is not None:
            valid.append(int(index))
            targets.append(target)
    indices = np.asarray(valid, dtype=np.int64)
    if len(indices) < 100:
        raise RuntimeError(f"only {len(indices)} horizon-matched rows available")
    endpoint = predicted_terminal[indices]
    action = actions[indices]
    target = np.asarray(targets, dtype=np.float64)
    roots = np.unique(episode[indices])
    rng = np.random.default_rng(args.seed)
    rng.shuffle(roots)
    n_train = max(1, int(round(0.60 * len(roots))))
    n_cal = max(1, int(round(0.20 * len(roots))))
    masks = {
        "train": np.isin(episode[indices], roots[:n_train]),
        "calibration": np.isin(episode[indices], roots[n_train:n_train + n_cal]),
        "test": np.isin(episode[indices], roots[n_train + n_cal:]),
    }
    features = {
        "predicted_terminal": endpoint,
        "predicted_terminal_action": np.column_stack([endpoint, action]),
    }
    models = {}
    metrics = {}
    for name, values in features.items():
        model = _fit(values[masks["train"]], target[masks["train"]], args.ridge)
        models[name] = model
        metrics[name] = {split: _metrics(model, values[mask], target[mask]) for split, mask in masks.items()}
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    coef, mean, scale = models["predicted_terminal_action"]
    np.savez_compressed(
        output.with_suffix(".npz"),
        coefficient=coef.astype(np.float32),
        feature_mean=mean.astype(np.float32),
        feature_scale=scale.astype(np.float32),
        feature_names=np.asarray(["predicted_terminal_latent", "action_block"], dtype=object),
    )
    payload = {
        "schema": "tworoom_action_conditioned_readout.v1",
        "input": str(args.latents),
        "horizon": int(args.horizon),
        "ridge": float(args.ridge),
        "seed": int(args.seed),
        "split": "episode-disjoint 60/20/20 train/calibration/test",
        "rows_after_horizon_join": int(len(indices)),
        "metrics": metrics,
        "selected_runtime_model": "predicted_terminal_action",
        "readout": str(output.with_suffix(".npz")),
        "claim_boundary": "Offline action-conditioned physical successor diagnostic; not causal identification or online planning evidence.",
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
