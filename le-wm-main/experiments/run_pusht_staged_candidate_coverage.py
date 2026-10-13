"""Audit PushT candidate coverage separately from latent prediction.

This runner is deliberately simulator-state based.  It answers a narrow
question that the previous MPC results could not answer: did the candidate
bank contain a useful approach--push trajectory at the state presented to the
predictor?  It compares the old first-motion bootstrap with a settled-contact
bootstrap, and uniform, legacy, staged, and simulator-oracle candidates on
the same held-out states.  It is a coverage diagnostic, not a visual-policy
or frozen-LeWM success claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_repaired_predictor_mpc import (
    proposal_sequences,
    rollout_sequence,
    staged_proposal_sequences,
    stable_contact_bootstrap,
    task_state_cost,
)
from experiments.run_pusht_contact_predicted_latent_adapter import sample_cases
from experiments.run_pusht_learned_physical_dynamics import state_features

gym.logger.min_level = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--held-out", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--action-block", type=int, default=2)
    parser.add_argument("--bootstrap-steps", type=int, default=80)
    parser.add_argument("--oracle-iterations", type=int, default=2)
    parser.add_argument("--oracle-population", type=int, default=64)
    parser.add_argument("--seed", type=int, default=8421)
    return parser.parse_args()


def angle_error(state: np.ndarray, goal: np.ndarray) -> float:
    return float(abs((state[4] - goal[4] + np.pi) % (2.0 * np.pi) - np.pi))


def success(state: np.ndarray, goal: np.ndarray) -> bool:
    return bool(
        np.linalg.norm(np.asarray(state)[:4] - np.asarray(goal)[:4]) < 20.0
        and angle_error(state, goal) < np.pi / 9.0
    )


def evaluate_candidates(
    env: gym.Env, case: dict[str, Any], candidates: np.ndarray
) -> dict[str, Any]:
    actual_costs: list[float] = []
    block_progress: list[float] = []
    successes: list[bool] = []
    block_pose_successes: list[bool] = []
    block_xy_distances: list[float] = []
    initial_block_distance = float(
        np.linalg.norm(np.asarray(case["state"])[2:4] - np.asarray(case["goal"])[2:4])
    )
    for candidate in candidates:
        states, _ = rollout_sequence(env, case, candidate)
        final = np.asarray(states[-1], dtype=np.float64)
        actual_costs.append(task_state_cost(state_features(final), case["goal"]))
        block_distance = float(np.linalg.norm(final[2:4] - case["goal"][2:4]))
        block_progress.append(initial_block_distance - block_distance)
        block_xy_distances.append(block_distance)
        successes.append(success(final, case["goal"]))
        block_pose_successes.append(
            bool(block_distance < 20.0 and angle_error(final, case["goal"]) < np.pi / 9.0)
        )
    costs = np.asarray(actual_costs, dtype=np.float64)
    progress = np.asarray(block_progress, dtype=np.float64)
    best = int(np.argmin(costs))
    return {
        "candidate_count": int(len(candidates)),
        "oracle_best_task_cost": float(costs[best]),
        "oracle_best_block_progress": float(progress[np.argmax(progress)]),
        "oracle_success": bool(np.any(successes)),
        "block_pose_success": bool(np.any(block_pose_successes)),
        "best_block_xy_distance": float(np.min(block_xy_distances)),
        "success_count": int(np.sum(successes)),
        "cost_p10": float(np.quantile(costs, 0.10)),
        "cost_median": float(np.median(costs)),
        "best_index": best,
        "all_finite": bool(np.isfinite(costs).all() and np.isfinite(progress).all()),
    }


def oracle_candidates(
    env: gym.Env,
    case: dict[str, Any],
    count: int,
    horizon: int,
    action_block: int,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Use simulator rollouts to optimize a candidate bank as an upper bound."""
    blocks = int(np.ceil(horizon / action_block))
    rng = np.random.default_rng(seed)
    mean = np.zeros((blocks, 2), dtype=np.float64)
    std = np.full((blocks, 2), 0.8, dtype=np.float64)
    total_rollouts = 0
    for _ in range(iterations):
        bank = np.clip(
            mean[None] + std[None] * rng.normal(size=(count, blocks, 2)),
            -1.0,
            1.0,
        ).astype(np.float32)
        costs = []
        for candidate in bank:
            states, _ = rollout_sequence(
                env, case, np.repeat(candidate, action_block, axis=0)[:horizon]
            )
            costs.append(task_state_cost(state_features(states[-1]), case["goal"]))
        total_rollouts += len(bank)
        elite_count = max(8, count // 8)
        elite = bank[np.argsort(np.asarray(costs))[:elite_count]]
        elite = elite.reshape(elite_count, blocks, 2)
        mean = 0.35 * mean + 0.65 * elite.mean(axis=0)
        std = 0.35 * std + 0.65 * np.maximum(elite.std(axis=0), 0.05)
    final = np.clip(
        mean[None] + std[None] * rng.normal(size=(count, blocks, 2)), -1.0, 1.0
    ).astype(np.float32)
    return np.repeat(final, action_block, axis=1)[:, :horizon], total_rollouts


def run_group(
    env: gym.Env,
    cases: list[dict[str, Any]],
    generators: dict[str, Callable[[np.random.Generator, dict[str, Any]], np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in generators}
    for index, case in enumerate(cases):
        for name, generator in generators.items():
            candidate_set = generator(np.random.default_rng(args.seed + 1000 * index + len(name)), case)
            metrics = evaluate_candidates(env, case, candidate_set)
            metrics.update({
                "case": index,
                "bootstrap_stable_contact": bool(case.get("stable_contact", False)),
                "agent_block_distance": float(case.get("agent_block_distance", np.nan)),
                "agent_speed": float(case.get("agent_speed", np.nan)),
            })
            rows[name].append(metrics)
    summary = {}
    for name, values in rows.items():
        summary[name] = {
            "n": len(values),
            "oracle_best_task_cost_mean": float(np.mean([v["oracle_best_task_cost"] for v in values])),
            "oracle_best_task_cost_median": float(np.median([v["oracle_best_task_cost"] for v in values])),
            "oracle_success_rate": float(np.mean([v["oracle_success"] for v in values])),
            "block_pose_success_rate": float(np.mean([v["block_pose_success"] for v in values])),
            "mean_best_block_xy_distance": float(np.mean([v["best_block_xy_distance"] for v in values])),
            "mean_best_block_progress": float(np.mean([v["oracle_best_block_progress"] for v in values])),
            "all_finite": bool(all(v["all_finite"] for v in values)),
        }
    return {"summary": summary, "rows": rows}


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        first_motion = sample_cases(env, args.held_out, args.seed + 100000, args.bootstrap_steps)
        stable_cases = [
            stable_contact_bootstrap(env, args.seed + 100000 + index, args.bootstrap_steps)
            for index in range(args.held_out)
        ]
        def make_generators() -> dict[str, Callable[[np.random.Generator, dict[str, Any]], np.ndarray]]:
            return {
                "uniform": lambda rng, case: rng.uniform(
                    -1.0, 1.0, size=(args.candidates, args.horizon, 2)
                ).astype(np.float32),
                "legacy_guided": lambda rng, case: proposal_sequences(
                    rng, case["state"], case["goal"], args.candidates,
                    args.horizon, args.action_block, 0.22,
                ),
                "staged": lambda rng, case: staged_proposal_sequences(
                    rng, case["state"], case["goal"], args.candidates,
                    args.horizon, args.action_block, 0.12,
                ),
            }
        first_report = run_group(env, first_motion, make_generators(), args)
        stable_report = run_group(env, stable_cases, make_generators(), args)
        oracle_rows = []
        for index, case in enumerate(stable_cases):
            candidates, rollouts = oracle_candidates(
                env, case, args.oracle_population, args.horizon,
                args.action_block, args.oracle_iterations, args.seed + 500000 + index,
            )
            metrics = evaluate_candidates(env, case, candidates)
            metrics.update({"case": index, "simulator_rollouts": rollouts})
            oracle_rows.append(metrics)
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "held_out": args.held_out,
            "candidate_count": args.candidates,
            "horizon": args.horizon,
            "action_block": args.action_block,
            "bootstrap_comparison": "first block displacement versus settled contact",
            "success_definition": "PushT position error < 20 and angle error < pi/9",
            "oracle_definition": "simulator-in-the-loop CEM upper-bound diagnostic; not available to LeWM",
        },
        "first_motion_bootstrap": first_report,
        "stable_contact_bootstrap": stable_report,
        "simulator_oracle": {
            "rows": oracle_rows,
            "summary": {
                "oracle_best_task_cost_mean": float(np.mean([r["oracle_best_task_cost"] for r in oracle_rows])),
                "oracle_success_rate": float(np.mean([r["oracle_success"] for r in oracle_rows])),
                "block_pose_success_rate": float(np.mean([r["block_pose_success"] for r in oracle_rows])),
                "mean_best_block_xy_distance": float(np.mean([r["best_block_xy_distance"] for r in oracle_rows])),
                "all_finite": bool(all(r["all_finite"] for r in oracle_rows)),
            },
        },
        "claim_boundary": (
            "Privileged simulator-state candidate-coverage audit. It diagnoses "
            "bootstrap and proposal support; it is not evidence that the frozen "
            "visual latent supports causal counterfactuals or online task success."
        ),
    }
    (output / "staged_candidate_coverage.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# PushT staged candidate coverage",
        "",
        "| bootstrap | candidate set | oracle best task cost | oracle success | block progress |",
        "|---|---|---:|---:|---:|",
    ]
    for bootstrap_name, value in (("first-motion", first_report), ("stable-contact", stable_report)):
        for name, summary in value["summary"].items():
            lines.append(
                "| %s | %s | %.3f | %.3f | %.3f |"
                % (bootstrap_name, name, summary["oracle_best_task_cost_mean"],
                   summary["oracle_success_rate"], summary["mean_best_block_progress"])
            )
    oracle = report["simulator_oracle"]["summary"]
    lines.append(
        "| stable-contact | simulator oracle | %.3f | %.3f | n/a |"
        % (oracle["oracle_best_task_cost_mean"], oracle["oracle_success_rate"])
    )
    lines += ["", report["claim_boundary"], ""]
    (output / "staged_candidate_coverage.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "summary": report["stable_contact_bootstrap"]["summary"], "oracle": oracle}, indent=2))


if __name__ == "__main__":
    main()
