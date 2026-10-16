"""Shared multi-stage action-sequence ranking audit for TwoRoom LeWM.

This separates candidate-support failure from frozen-model utility failure.
The same deterministic sequence bank is scored by the RGB-only frozen model
and replayed in the simulator for held-out labels.  The simulator is never
used by the model scorer.
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


def _sequence_bank(seed: int, count: int, horizon: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bank = rng.uniform(-0.75, 0.75, size=(count, horizon, 2)).astype(np.float32)
    # Reserve interpretable controls in the shared bank before random
    # multi-stage sequences: hold, axis move, and a turn-then-translate pulse.
    if count >= 1:
        bank[0] = 0.0
    if count >= 5:
        bank[1:] = bank[1:]
        bank[1, :, :] = np.asarray([0.5, 0.0], dtype=np.float32)
        bank[2, :, :] = np.asarray([-0.5, 0.0], dtype=np.float32)
        bank[3, :, :] = np.asarray([0.0, 0.5], dtype=np.float32)
        bank[4, :, :] = np.asarray([0.0, -0.5], dtype=np.float32)
    if count >= 6:
        bank[5, : max(horizon // 2, 1)] = np.asarray([0.5, 0.0], dtype=np.float32)
        bank[5, max(horizon // 2, 1):] = np.asarray([0.0, 0.5], dtype=np.float32)
    return bank


def _normalize_bank(raw_bank: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    if raw_bank.ndim != 3 or raw_bank.shape[-1] != 2:
        raise ValueError(f"raw bank must have shape [N,H,2], got {raw_bank.shape}")
    blocks = np.repeat(raw_bank[..., None, :], 5, axis=2).reshape(
        raw_bank.shape[0], raw_bank.shape[1], 10
    )
    normalized = (blocks - mean.reshape(1, 1, -1)) / np.maximum(
        std.reshape(1, 1, -1), 1e-6
    )
    return torch.as_tensor(normalized, dtype=torch.float32)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/tworoom/lewm")
    parser.add_argument(
        "--adapter-artifact", type=Path,
        default=Path("results/live_cross_environment_prediction_20261015/tworoom_cuda/cross_environment_prediction.json"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261130)
    parser.add_argument("--bank-seed", type=int, default=20261131)
    parser.add_argument("--roots", type=int, default=16)
    parser.add_argument("--bank-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--replay-steps", type=int, default=25)
    parser.add_argument("--history-len", type=int, default=1)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    action_mean, action_std = _load_action_stats(args.adapter_artifact, "swm/TwoRoom-v1")
    normalizer = ActionNormalizer(action_mean, action_std)
    raw_bank = _sequence_bank(args.bank_seed, args.bank_size, args.horizon)
    normalized_bank = _normalize_bank(raw_bank, action_mean, action_std)
    model = load_compatible_pretrained(args.policy, device=args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    preprocessor = _Preprocessor(normalizer)
    env = TwoRoomGoalAliasWrapper(gym.make("swm/TwoRoom-v1", render_mode="rgb_array"))
    rows = []
    try:
        for root_index in range(args.roots):
            seed = int(args.seed + root_index)
            _, info = env.reset(seed=seed)
            current = np.asarray(env.render(), dtype=np.uint8).copy()
            goal = np.asarray(info["goal"], dtype=np.uint8).copy()
            model_cost = _model_costs(
                model, preprocessor, current, goal, normalizer,
                normalized_bank, args.device, args.history_len,
            )
            true_cost = np.asarray([
                _rollout_candidate(
                    env, seed, candidate, args.replay_steps, args.action_block,
                )[0]
                for candidate in raw_bank
            ], dtype=np.float64)
            predicted_best = int(np.argmin(model_cost))
            true_order = np.argsort(true_cost, kind="stable")
            rows.append({
                "seed": seed,
                "predicted_best": predicted_best,
                "true_best": int(true_order[0]),
                "true_rank_of_predicted_best": int(np.flatnonzero(true_order == predicted_best)[0] + 1),
                "top1_hit": int(predicted_best == int(true_order[0])),
                "top4_hit": int(predicted_best in set(true_order[:4].tolist())),
                "regret": float(true_cost[predicted_best] - true_cost[true_order[0]]),
                "spearman": _spearman(model_cost, true_cost),
                "model_cost": model_cost.tolist(),
                "true_cost": true_cost.tolist(),
            })
    finally:
        env.close()
    top1 = np.asarray([row["top1_hit"] for row in rows], dtype=np.float64)
    top4 = np.asarray([row["top4_hit"] for row in rows], dtype=np.float64)
    ranks = np.asarray([row["true_rank_of_predicted_best"] for row in rows], dtype=np.float64)
    regrets = np.asarray([row["regret"] for row in rows], dtype=np.float64)
    spearman = np.asarray([row["spearman"] for row in rows], dtype=np.float64)
    payload = {
        "schema": "tworoom_sequence_candidate_ranking.v1",
        "protocol": {
            "roots": "seed-disjoint root replay",
            "candidate_bank": "shared deterministic multi-stage macro-action sequences",
            "candidate_count": int(len(raw_bank)),
            "horizon": int(args.horizon),
            "action_block": int(args.action_block),
            "replay_steps": int(args.replay_steps),
            "model_input": "RGB current frame and RGB goal only",
            "history_len": int(args.history_len),
            "simulator_use": "held-out labels only; never passed to model ranking",
            "checkpoint_frozen": True,
            "device": args.device,
            "bank_seed": int(args.bank_seed),
        },
        "roots": rows,
        "summary": {
            "n": int(len(rows)),
            "top1_accuracy": float(top1.mean()),
            "top4_accuracy": float(top4.mean()),
            "mean_true_rank_of_predicted_best": float(ranks.mean()),
            "mean_regret": float(regrets.mean()),
            "median_regret": float(np.median(regrets)),
            "mean_spearman": float(spearman.mean()),
        },
        "candidate_bank": raw_bank.tolist(),
        "claim_boundary": "Shared-candidate RGB sequence ranking diagnostic with privileged offline labels; not online planning or a universal transfer certificate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
