"""Root-disjoint latent/action utility ranker audit for TwoRoom LeWM.

The encoder and predictor remain frozen.  A small ridge readout is fitted on
calibration roots only and predicts the held-out replay distance for each
shared candidate.  This tests whether the latent still contains enough joint
state-goal-action-effect information for a task readout; it is not an online
planning claim until a fresh live consequence test is passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.audit_tworoom_shared_candidate_ranking import (
    _Preprocessor,
    _candidate_bank,
    _model_costs,
    _rollout_candidate,
    _spearman,
)
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_tworoom_planning import (
    TwoRoomGoalAliasWrapper,
    _image_transform,
    _load_action_stats,
)


@torch.inference_mode()
def _model_latent_features(
    model: torch.nn.Module,
    preprocessor: _Preprocessor,
    current: np.ndarray,
    goal: np.ndarray,
    candidates: torch.Tensor,
    device: str,
    history_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if history_len <= 0:
        raise ValueError("history_len must be positive")
    raw = {
        "pixels": np.repeat(current[None, None], history_len, axis=1),
        "goal": goal[None, None],
        "action": np.zeros((1, history_len, 2), dtype=np.float32),
    }
    prepared = preprocessor._prepare_info(raw)
    model_info = {}
    bank_size = int(candidates.shape[0])
    for key in ("pixels", "goal", "action"):
        value = prepared[key].to(device)
        model_info[key] = value[:, None].expand(1, bank_size, *value.shape[1:]).contiguous()
    candidate_tensor = candidates.to(device).unsqueeze(0)
    costs = model.get_cost(model_info, candidate_tensor).squeeze(0)
    predicted = model_info["predicted_emb"][0, :, -1, :]
    goal_embedding = model_info["goal_emb"][0, -1, :].unsqueeze(0).expand_as(predicted)
    current_embedding = model_info["emb"][0, :, -1, :]
    return (
        costs.detach().cpu().numpy(),
        predicted.detach().cpu().numpy(),
        goal_embedding.detach().cpu().numpy(),
        current_embedding.detach().cpu().numpy(),
    )


def _features(
    costs: np.ndarray,
    predicted: np.ndarray,
    goal: np.ndarray,
    current: np.ndarray,
    raw_actions: np.ndarray,
) -> np.ndarray:
    cost_scale = max(float(costs.std()), 1e-8)
    z_cost = (costs - costs.mean()) / cost_scale
    delta = predicted - goal
    context = current - goal
    return np.column_stack([
        z_cost,
        raw_actions.reshape(len(raw_actions), -1),
        delta,
        delta * delta,
        delta * context,
    ])


def _fit_ridge(features: np.ndarray, target: np.ndarray, ridge: float):
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    normalized = (features - mean) / scale
    design = np.column_stack([normalized, np.ones(len(normalized))])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return coef, mean, scale


def _predict(features: np.ndarray, fit) -> np.ndarray:
    coef, mean, scale = fit
    design = np.column_stack([
        (features - mean) / scale,
        np.ones(len(features)),
    ])
    return design @ coef


def _evaluate(rows: list[dict], indices: np.ndarray, predictions: dict[int, np.ndarray]) -> dict:
    top1, top4, ranks, regrets = [], [], [], []
    for index in indices.tolist():
        row = rows[index]
        scores = predictions[index]
        true = np.asarray(row["true_cost"], dtype=np.float64)
        predicted_best = int(np.argmin(scores))
        order = np.argsort(true, kind="stable")
        top1.append(int(predicted_best == int(order[0])))
        top4.append(int(predicted_best in set(order[:4].tolist())))
        ranks.append(int(np.flatnonzero(order == predicted_best)[0] + 1))
        regrets.append(float(true[predicted_best] - true[order[0]]))
    return {
        "n": int(len(indices)),
        "top1_accuracy": float(np.mean(top1)),
        "top4_accuracy": float(np.mean(top4)),
        "mean_true_rank_of_predicted_best": float(np.mean(ranks)),
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "top1_hits": int(np.sum(top1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/tworoom/lewm")
    parser.add_argument(
        "--adapter-artifact", type=Path,
        default=Path("results/live_cross_environment_prediction_20261015/tworoom_cuda/cross_environment_prediction.json"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261200)
    parser.add_argument("--roots", type=int, default=32)
    parser.add_argument("--calibration-roots", type=int, default=16)
    parser.add_argument("--replay-steps", type=int, default=25)
    parser.add_argument("--history-len", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=100.0)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.calibration_roots >= args.roots:
        raise ValueError("calibration roots must be smaller than total roots")
    action_mean, action_std = _load_action_stats(args.adapter_artifact, "swm/TwoRoom-v1")
    normalizer = ActionNormalizer(action_mean, action_std)
    raw_bank, normalized_bank = _candidate_bank(action_mean, action_std, horizon=5, action_block=5)
    raw_actions = np.asarray(raw_bank, dtype=np.float64)
    model = load_compatible_pretrained(args.policy, device=args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    preprocessor = _Preprocessor(normalizer)
    env = TwoRoomGoalAliasWrapper(gym.make("swm/TwoRoom-v1", render_mode="rgb_array"))
    rows: list[dict] = []
    features_by_root: list[np.ndarray] = []
    try:
        for root_index in range(args.roots):
            seed = int(args.seed + root_index)
            _, info = env.reset(seed=seed)
            current = np.asarray(env.render(), dtype=np.uint8).copy()
            goal = np.asarray(info["goal"], dtype=np.uint8).copy()
            costs, predicted, goal_emb, current_emb = _model_latent_features(
                model, preprocessor, current, goal, normalized_bank,
                args.device, args.history_len,
            )
            true_cost = np.asarray([
                _rollout_candidate(env, seed, action, args.replay_steps)[0]
                for action in raw_bank
            ], dtype=np.float64)
            features_by_root.append(_features(costs, predicted, goal_emb, current_emb, raw_actions))
            rows.append({
                "seed": seed,
                "true_cost": true_cost.tolist(),
                "raw_model_cost": costs.tolist(),
                "raw_spearman": _spearman(costs, true_cost),
            })
    finally:
        env.close()

    calibration = np.arange(args.calibration_roots, dtype=np.int64)
    held_out = np.arange(args.calibration_roots, len(rows), dtype=np.int64)
    fit = _fit_ridge(
        np.concatenate([features_by_root[i] for i in calibration], axis=0),
        np.concatenate([np.asarray(rows[i]["true_cost"], dtype=np.float64) for i in calibration]),
        args.ridge,
    )
    ranker_scores = {
        i: _predict(features_by_root[i], fit)
        for i in range(len(rows))
    }
    raw_scores = {
        i: np.asarray(rows[i]["raw_model_cost"], dtype=np.float64)
        for i in range(len(rows))
    }
    summary = {
        "calibration_raw": _evaluate(rows, calibration, raw_scores),
        "calibration_latent_action_ranker": _evaluate(rows, calibration, ranker_scores),
        "held_out_raw": _evaluate(rows, held_out, raw_scores),
        "held_out_latent_action_ranker": _evaluate(rows, held_out, ranker_scores),
    }
    summary["held_out_delta_ranker_minus_raw_top1"] = (
        summary["held_out_latent_action_ranker"]["top1_accuracy"]
        - summary["held_out_raw"]["top1_accuracy"]
    )
    summary["held_out_delta_ranker_minus_raw_regret"] = (
        summary["held_out_latent_action_ranker"]["mean_regret"]
        - summary["held_out_raw"]["mean_regret"]
    )
    output = {
        "schema": "tworoom_latent_action_ranker.v1",
        "status": "exploratory_root_disjoint",
        "protocol": {
            "roots": int(args.roots),
            "calibration_roots": [int(rows[i]["seed"]) for i in calibration],
            "held_out_roots": [int(rows[i]["seed"]) for i in held_out],
            "candidate_bank": "shared constant macro-action bank",
            "feature_family": "normalized frozen model cost, raw action, predicted-goal latent delta, delta square, delta-context interaction",
            "ridge": float(args.ridge),
            "history_len": int(args.history_len),
            "replay_steps": int(args.replay_steps),
            "simulator_labels": "calibration/evaluation only",
            "device": args.device,
        },
        "summary": summary,
        "claim_boundary": "Frozen-latent task-readout diagnostic; no online planning, causal identification, or universal transfer claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranker_artifact = args.output.with_suffix(".npz")
    coefficient, feature_mean, feature_scale = fit
    np.savez_compressed(
        ranker_artifact,
        coefficient=np.asarray(coefficient, dtype=np.float32),
        feature_mean=np.asarray(feature_mean, dtype=np.float32),
        feature_scale=np.asarray(feature_scale, dtype=np.float32),
        feature_names=np.asarray(
            ["normalized_base_cost", "raw_macro_action_xy", "latent_delta",
             "latent_delta_square", "latent_delta_context"],
            dtype=object,
        ),
        calibration_roots=np.asarray([rows[i]["seed"] for i in calibration], dtype=np.int64),
    )
    output["ranker_artifact"] = str(ranker_artifact)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
