"""Audit the held-out action-ranker interface without changing experiment artifacts.

This diagnostic reconstructs the exact root-disjoint validation intervention bank,
then separates three possible causes of a failed ranker: label/support ordering,
feature-interface mismatch, and model fit.  It is intentionally read-only and
does not promote any controller variant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from run_rgb_local_effect_adapter import (
    attach_targets,
    balanced_source_cases,
    block_pose,
    frozen_features,
    load_base,
    make_local_effect_dataset,
    pose_cost,
    predict_action_cost,
    rgb_geometry_ranker_features,
    terminal_pose_cases,
    theta_from_pose,
    action_bank,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    payload = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
    model, _normalizer, _ = load_base(Path(args.base_checkpoint).resolve(), args.device)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=80)
    try:
        actions = action_bank(16, [0.35, 0.60, 0.85])
        roots = balanced_source_cases(env, 24, 8, 9173 + 10_000, 0.6, 48)
        roots.extend(terminal_pose_cases(env, 8, 9173 + 410_000, position_jitter=16.0))
        data = attach_targets(
            make_local_effect_dataset(env, roots, 64, actions, 1, "aligned_state_goal"),
            actions,
        )
        features = frozen_features(model, data, 64, args.device)
        current = model.encoder.features(
            torch.as_tensor(data["image"][[i * len(actions) for i in range(len(roots))]], device=args.device)
        ).cpu().numpy()
        previous = model.encoder.features(
            torch.as_tensor(data["previous_image"][[i * len(actions) for i in range(len(roots))]], device=args.device)
        ).cpu().numpy()
        goal = model.encoder.features(
            torch.as_tensor(data["goal_image"][[i * len(actions) for i in range(len(roots))]], device=args.device)
        ).cpu().numpy()
        goal_states = np.asarray([roots[i][1][2:6] for i in range(len(roots))], dtype=np.float32)
        current_angles = np.asarray([roots[i][0][4] for i in range(len(roots))], dtype=np.float32)
        goal_rows = np.repeat(goal_states, len(actions), axis=0)
        angle_rows = np.repeat(current_angles, len(actions), axis=0)
        rank_features = rgb_geometry_ranker_features(
            np.asarray(data["image"]), np.asarray(data["previous_image"]),
            np.asarray(data["goal_image"]), data["action"], angle_rows, goal_rows,
        )
        ranker = __import__("run_rgb_local_effect_adapter", fromlist=["ActionCostRanker"]).ActionCostRanker(
            rank_features.shape[1], int(payload["hidden_dim"]),
        ).to(args.device)
        ranker.load_state_dict(payload["action_cost_ranker"])
        ranker.eval()
        predicted = predict_action_cost(
            ranker, rank_features, payload["action_cost_ranker_training"], args.device,
        )
        row_true = np.asarray([
            [row[2], row[3], np.arctan2(row[4], row[5])] for row in data["next_state"]
        ], dtype=np.float32)
        row_goal = np.asarray([
            [row[0], row[1], np.arctan2(row[2], row[3])] for row in goal_rows
        ], dtype=np.float32)
        true_cost = (
            np.linalg.norm(row_true[:, :2] - row_goal[:, :2], axis=1) / 20.0
            + 10.0 * np.abs(
                (row_true[:, 2] - row_goal[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
            )
        ).astype(np.float32)
        rows = []
        hits = []
        for root_id in range(len(roots)):
            indices = np.flatnonzero(data["root_id"] == root_id)
            selected = int(indices[np.argmin(predicted[indices])])
            oracle = int(indices[np.argmin(true_cost[indices])])
            hits.append(float(selected == oracle))
            rows.append({
                "root": root_id,
                "contact": bool(data["contact_root"][indices[0]]),
                "predicted_action_index": int(np.argmin(predicted[indices])),
                "oracle_action_index": int(np.argmin(true_cost[indices])),
                "predicted_cost": float(predicted[selected]),
                "oracle_cost": float(true_cost[oracle]),
                "selected_true_cost": float(true_cost[selected]),
                "regret": float(true_cost[selected] - true_cost[oracle]),
            })
        output = {
            "roots": len(roots),
            "actions_per_root": len(actions),
            "feature_dim": int(rank_features.shape[1]),
            "train_interface": payload["action_cost_ranker_training"].get("regression_target"),
            "oracle_current_true_goal_top1": float(np.mean(hits)),
            "mean_regret": float(np.mean([row["regret"] for row in rows])),
            "contact_top1": float(np.mean([hit for hit, row in zip(hits, rows) if row["contact"]])),
            "rows": rows,
        }
        print(json.dumps(output, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
