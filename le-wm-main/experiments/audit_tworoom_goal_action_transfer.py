"""Root-disjoint goal/action transfer audit for the TwoRoom LeWM export.

The full latent export contains overlapping trajectory windows rather than a
counterfactual action bank.  This audit therefore does not claim causal action
ranking or online planning.  It uses only adjacent rows from the same episode
to recover a held-out one-step physical successor, fits fixed ridge readouts
on disjoint training episodes, and asks whether goal/action conditioning makes
the learned latent effect more decision-relevant.

The endpoint ``goals`` field is the final state of the exported trajectory
window, not an externally specified task goal.  The protocol records this
boundary explicitly and reports it as an endpoint-goal transfer diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


MODEL_FEATURES = {
    "state_action": ("state", "action"),
    "latent_action": ("latent", "action"),
    "latent_goal_action": ("latent", "goal", "action"),
    "predictor_effect_goal_action": ("predicted_dynamics", "goal", "action"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def _load(path: Path, max_samples: int, seed: int) -> dict[str, np.ndarray]:
    required = {
        "latents", "predicted_dynamics", "actions", "goals", "physical_proprio",
        "episode_id", "timestep",
    }
    with np.load(path) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path} is missing required arrays: {missing}")
        arrays = {key: np.asarray(data[key]) for key in required}
    if max_samples and len(arrays["latents"]) > max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(arrays["latents"]), max_samples, replace=False))
        arrays = {key: value[indices] for key, value in arrays.items()}
    return arrays


def _deduplicate_and_join_next(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    episode = data["episode_id"].reshape(-1).astype(np.int64)
    timestep = data["timestep"].reshape(-1).astype(np.int64)
    order = np.lexsort((timestep, episode))
    ordered_keys = np.stack([episode[order], timestep[order]], axis=1)
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = np.any(ordered_keys[1:] != ordered_keys[:-1], axis=1)
    chosen = order[keep]

    # The exported physical field is the current state.  Build a lookup only
    # from the de-duplicated rows, so repeated overlapping windows cannot leak
    # a duplicate target into either split.
    current = data["physical_proprio"].reshape(len(episode), -1).astype(np.float64)
    states = data["states"].reshape(len(episode), -1).astype(np.float64) if "states" in data else current
    next_map = {
        (int(ep), int(ts)): current[index]
        for index, ep, ts in zip(chosen, episode[chosen], timestep[chosen])
    }
    valid: list[int] = []
    next_states: list[np.ndarray] = []
    for index in chosen:
        key = (int(episode[index]), int(timestep[index]) + 1)
        target = next_map.get(key)
        if target is not None:
            valid.append(int(index))
            next_states.append(target)
    if len(valid) < 100:
        raise RuntimeError(f"only {len(valid)} adjacent rows available")
    indices = np.asarray(valid, dtype=np.int64)
    out = {
        "latents": data["latents"][indices].astype(np.float64),
        "predicted_dynamics": data["predicted_dynamics"][indices].astype(np.float64),
        "actions": data["actions"][indices, :2].astype(np.float64),
        "goals": data["goals"][indices].reshape(len(indices), -1).astype(np.float64),
        "state": states[indices, :2].astype(np.float64),
        "next_state": np.asarray(next_states, dtype=np.float64)[:, :2],
        "episode_id": episode[indices],
        "timestep": timestep[indices],
    }
    return out


def _split_by_episode(data: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    episodes = np.unique(data["episode_id"])
    rng = np.random.default_rng(seed)
    shuffled = episodes.copy()
    rng.shuffle(shuffled)
    n_train = max(1, int(round(0.60 * len(shuffled))))
    n_cal = max(1, int(round(0.20 * len(shuffled))))
    split = np.full(len(data["episode_id"]), "test", dtype="U5")
    split[np.isin(data["episode_id"], shuffled[:n_train])] = "train"
    split[np.isin(data["episode_id"], shuffled[n_train:n_train + n_cal])] = "cal"
    return {"train": split == "train", "cal": split == "cal", "test": split == "test"}


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    return mean, scale


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, scale = _standardize_fit(x)
    xs = (x - mean) / scale
    design = np.column_stack([xs, np.ones(len(xs))])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return coefficients, mean, scale


def _predict(model: tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    coefficients, mean, scale = model
    xs = (x - mean) / scale
    return np.column_stack([xs, np.ones(len(xs))]) @ coefficients


def _feature_matrix(data: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    columns = [
        data["actions"] if name == "action" else
        data["latents"] if name == "latent" else
        data["goals"] if name == "goal" else
        data[name]
        for name in names
    ]
    return np.column_stack(columns).astype(np.float64)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _score_model(
    model: tuple[np.ndarray, np.ndarray, np.ndarray],
    data: dict[str, np.ndarray],
    names: tuple[str, ...],
    indices: np.ndarray,
) -> dict[str, Any]:
    subset = {key: value[indices] for key, value in data.items()}
    predicted_delta = _predict(model, _feature_matrix(subset, names))
    true_delta = subset["next_state"] - subset["state"]
    goal_delta = subset["goals"][:, :2] - subset["state"]
    current_distance = np.linalg.norm(goal_delta, axis=1)
    next_distance = np.linalg.norm(subset["goals"][:, :2] - subset["next_state"], axis=1)
    true_progress = current_distance - next_distance
    predicted_next = subset["state"] + predicted_delta
    predicted_progress = current_distance - np.linalg.norm(subset["goals"][:, :2] - predicted_next, axis=1)

    zero_action = subset["actions"].copy()
    zero_action[:] = 0.0
    zero_features = dict(subset)
    zero_features["actions"] = zero_action
    predicted_zero = _predict(model, _feature_matrix(zero_features, names))
    zero_next = subset["state"] + predicted_zero
    predicted_action_contrast = (
        np.linalg.norm(subset["goals"][:, :2] - zero_next, axis=1)
        - np.linalg.norm(subset["goals"][:, :2] - predicted_next, axis=1)
    )
    return {
        "n": int(len(indices)),
        "next_delta_rmse": float(np.sqrt(np.mean(np.square(predicted_delta - true_delta)))),
        "goal_progress_rmse": float(np.sqrt(np.mean(np.square(predicted_progress - true_progress)))),
        "goal_progress_sign_accuracy": float(np.mean((predicted_progress > 0) == (true_progress > 0))),
        "goal_progress_correlation": _safe_corr(predicted_progress, true_progress),
        "action_contrast_mean": float(np.mean(predicted_action_contrast)),
        "action_contrast_positive_fraction": float(np.mean(predicted_action_contrast > 0)),
        "true_progress_mean": float(np.mean(true_progress)),
        "predicted_progress_mean": float(np.mean(predicted_progress)),
    }


def main() -> None:
    args = _parse_args()
    if not np.isfinite(args.ridge) or args.ridge <= 0:
        raise ValueError("--ridge must be positive and finite")
    raw = _load(args.latents, args.max_samples, args.seed)
    data = _deduplicate_and_join_next(raw)
    masks = _split_by_episode(data, args.seed)
    target = data["next_state"] - data["state"]
    records: dict[str, Any] = {}
    for model_name, names in MODEL_FEATURES.items():
        train_indices = np.flatnonzero(masks["train"])
        model = _fit_ridge(_feature_matrix(data, names)[train_indices], target[train_indices], args.ridge)
        records[model_name] = {
            "features": list(names),
            "calibration": _score_model(model, data, names, np.flatnonzero(masks["cal"])),
            "test": _score_model(model, data, names, np.flatnonzero(masks["test"])),
        }

    payload = {
        "schema": "tworoom_goal_action_transfer_audit.v1",
        "protocol": {
            "split": "episode-disjoint 60/20/20 train/calibration/test",
            "target": "adjacent physical_proprio next-state delta",
            "goal": "exported trajectory-window endpoint state, not external task goal",
            "fit": "fixed ridge linear readout fit on train episodes only",
            "duplicate_policy": "one row per episode/timestep before adjacent join",
            "runtime_state_access": False,
            "online_planning": False,
            "causal_action_ranking": False,
        },
        "input": str(args.latents),
        "seed": int(args.seed),
        "ridge": float(args.ridge),
        "rows_after_adjacent_join": int(len(data["state"])),
        "episodes": int(len(np.unique(data["episode_id"]))),
        "split_rows": {name: int(mask.sum()) for name, mask in masks.items()},
        "action_support": {
            "norm_p05": float(np.quantile(np.linalg.norm(data["actions"], axis=1), 0.05)),
            "norm_p50": float(np.quantile(np.linalg.norm(data["actions"], axis=1), 0.50)),
            "norm_p95": float(np.quantile(np.linalg.norm(data["actions"], axis=1), 0.95)),
        },
        "models": records,
        "claim_boundary": "Offline root-disjoint endpoint-goal transfer diagnostic; not causal action ranking, online planning, or universal world-model evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# TwoRoom goal/action transfer audit", "",
        "This is an episode-disjoint offline diagnostic. The endpoint goal is the final state of each exported window, not an external task goal.", "",
        "| model | split | delta RMSE | goal-progress RMSE | sign accuracy | progress corr |", "|---|---|---:|---:|---:|---:|",
    ]
    for model_name, record in records.items():
        for split in ("calibration", "test"):
            row = record[split]
            corr = "NA" if row["goal_progress_correlation"] is None else f"{row['goal_progress_correlation']:.4f}"
            lines.append(
                f"| {model_name} | {split} | {row['next_delta_rmse']:.4f} | {row['goal_progress_rmse']:.4f} | "
                f"{row['goal_progress_sign_accuracy']:.4f} | {corr} |"
            )
    lines += ["", "Claim boundary: this artifact does not establish causal action ranking, online planning, or universal transfer.", ""]
    args.output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
