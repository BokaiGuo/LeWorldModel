"""Disentangle learned-transition ranking error from planner-objective error."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_learned_physical_dynamics import (
    bootstrap, candidate_sequences, collect_transition_data, reset_case, state_features, step_case,
)
from experiments.run_pusht_pose_closure import fit_pose_dynamics, pose, rollout_pose


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--train", type=int, default=512)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--candidates", type=int, default=64)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--seed", type=int, default=10172)
    p.add_argument("--trees", type=int, default=96)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    output = np.empty(order.size, dtype=np.float64)
    output[order] = np.arange(order.size, dtype=np.float64)
    return output


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x, y = rank(x), rank(y)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    args = parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        train = collect_transition_data(env, args.train, args.seed, rng)
        model = fit_pose_dynamics(train["states"], train["actions"], train["next_states"], args.trees, args.seed)
        rows = []
        for episode in range(args.episodes):
            case = reset_case(env, args.seed + 10000 + episode)
            current = pose(state_features(case["state"])); goal = pose(state_features(case["goal"]))
            candidates = candidate_sequences(rng, args.candidates, 1)[:, 0, :]
            predicted, actual = [], []
            for action in candidates:
                predicted.append(rollout_pose(model, current, action, args.action_block))
                endpoint = step_case(env, case["state"], case["goal"], action, args.action_block)
                actual.append(pose(state_features(endpoint["state"])))
            predicted, actual = np.asarray(predicted), np.asarray(actual)
            predicted_block = np.linalg.norm(predicted[:, 2:4] - goal[2:4], axis=1)
            actual_block = np.linalg.norm(actual[:, 2:4] - goal[2:4], axis=1)
            predicted_task = np.linalg.norm(predicted[:, :4] - goal[:4], axis=1) + 20.0 * np.abs(
                np.arctan2(predicted[:, 4], predicted[:, 5]) - np.arctan2(goal[4], goal[5])
            )
            actual_task = np.linalg.norm(actual[:, :4] - goal[:4], axis=1) + 20.0 * np.abs(
                np.arctan2(actual[:, 4], actual[:, 5]) - np.arctan2(goal[4], goal[5])
            )
            rows.append({
                "episode": episode,
                "block_rank_spearman": correlation(predicted_block, actual_block),
                "task_rank_spearman": correlation(predicted_task, actual_task),
                "block_selected_regret": float(actual_block[int(np.argmin(predicted_block))] - np.min(actual_block)),
                "task_selected_regret": float(actual_task[int(np.argmin(predicted_task))] - np.min(actual_task)),
                "oracle_block_best": float(np.min(actual_block)),
                "oracle_task_best": float(np.min(actual_task)),
                "predicted_block_best_actual": float(actual_block[int(np.argmin(predicted_block))]),
                "predicted_task_best_actual": float(actual_block[int(np.argmin(predicted_task))]),
            })
    finally:
        env.close()
    finite = [row for row in rows if np.isfinite(row["block_rank_spearman"]) and np.isfinite(row["task_rank_spearman"])]
    metrics = {}
    for key in ("block_rank_spearman", "task_rank_spearman", "block_selected_regret", "task_selected_regret", "oracle_block_best", "oracle_task_best"):
        metrics[key] = bootstrap(np.asarray([row[key] for row in finite]), args.seed + len(key), args.bootstrap)
    report = {
        "protocol": {
            "environment": "swm/PushT-v1", "train_transitions": args.train,
            "episodes": args.episodes, "candidates_per_episode": args.candidates,
            "horizon": args.horizon, "action_block": args.action_block,
            "model": "pose-only ExtraTrees residual transition",
            "candidate_protocol": "same initial state, goal, and candidate actions replayed in simulator",
        },
        "metrics": metrics, "rows": rows,
        "sanity_checks": {"finite_rows": len(finite) == args.episodes, "all_finite": bool(finite)},
        "claim_boundary": "Candidate ranking diagnostic separates transition-model and objective errors; it is not online task success or causal-control evidence.",
    }
    (out / "candidate_rank_diagnostic.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = ["# PushT candidate-rank diagnostic", "", "| metric | estimate | 95% low | 95% high |", "|---|---:|---:|---:|"]
    for key, value in metrics.items():
        lines.append("| %s | %.3f | %.3f | %.3f |" % (key, value["estimate"], value["ci95_low"], value["ci95_high"]))
    lines += ["", "Claim boundary: candidate-ranking diagnosis, not online planning success.", ""]
    (out / "candidate_rank_diagnostic.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(out), "metrics": metrics, "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
