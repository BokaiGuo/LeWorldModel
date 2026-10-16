"""Fit and audit a root-disjoint latent-to-physical readout for TwoRoom."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _fit(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    xs = (x - mean) / scale
    design = np.column_stack([xs, np.ones(len(xs))])
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return coef, mean, scale


def _predict(model: tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    coef, mean, scale = model
    return np.column_stack([(x - mean) / scale, np.ones(len(x))]) @ coef


def _metrics(model, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = _predict(model, x)
    error = pred - y
    return {
        "n": int(len(x)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "x_rmse": float(np.sqrt(np.mean(np.square(error[:, 0])))),
        "y_rmse": float(np.sqrt(np.mean(np.square(error[:, 1])))),
        "target_std": float(np.mean(np.std(y, axis=0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--ridge", type=float, default=1e-2)
    args = parser.parse_args()
    with np.load(args.latents) as data:
        x = np.asarray(data["latents"], dtype=np.float64)
        y = np.asarray(data["physical_proprio"], dtype=np.float64).reshape(len(x), -1)[:, :2]
        episode = np.asarray(data["episode_id"]).reshape(-1).astype(np.int64)
    if x.shape[0] != y.shape[0] or len(np.unique(episode)) < 10:
        raise ValueError("latent, physical, and episode arrays are incompatible")
    rng = np.random.default_rng(args.seed)
    episodes = np.unique(episode)
    rng.shuffle(episodes)
    n_train = max(1, int(round(0.60 * len(episodes))))
    n_cal = max(1, int(round(0.20 * len(episodes))))
    masks = {
        "train": np.isin(episode, episodes[:n_train]),
        "calibration": np.isin(episode, episodes[n_train:n_train + n_cal]),
        "test": np.isin(episode, episodes[n_train + n_cal:]),
    }
    model = _fit(x[masks["train"]], y[masks["train"]], args.ridge)
    coef, mean, scale = model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        coefficient=coef.astype(np.float32),
        feature_mean=mean.astype(np.float32),
        feature_scale=scale.astype(np.float32),
    )
    payload = {
        "schema": "tworoom_physical_readout.v1",
        "input": str(args.latents),
        "ridge": float(args.ridge),
        "seed": int(args.seed),
        "split": "episode-disjoint 60/20/20 train/calibration/test",
        "rows": {name: int(mask.sum()) for name, mask in masks.items()},
        "episodes": {name: int(len(np.unique(episode[mask]))) for name, mask in masks.items()},
        "metrics": {name: _metrics(model, x[mask], y[mask]) for name, mask in masks.items()},
        "readout": str(args.output.with_suffix(".npz")),
        "claim_boundary": "Offline frozen latent-to-physical readout audit; not causal identification or online planning evidence.",
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
