"""Audit the first-step action ranking of the frozen RGB Cube planner.

This is an evaluator-only diagnostic for the Cube prediction-to-planning
failure.  The policy receives the same live RGB/goal/action-history inputs as
the closed-loop runner.  Privileged block positions and goal coordinates are
read only after reset/step to measure one-step task progress; they never enter
the model or the CEM cost.  The final CEM candidate costs and the selected
first action are recorded so that a prediction-level adapter gain can be
separated from a decision-ranking failure.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2 as transforms

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import (
    ActionAdaptedCostModel,
    ActionNormalizer,
    CubeGoalAliasWrapper,
    _image_transform,
    _load_adapter,
)


class FinalCEMTrace:
    """Keep the last CEM population for every solve without changing costs."""

    output_key = "final_cem_trace"

    def __init__(self, final_step: int, keep: int = 8):
        self.final_step = int(final_step)
        self.keep = int(keep)
        self.records: list[dict[str, Any]] = []
        self.history: list[list[Any]] = []

    def reset(self) -> None:
        # Solver reset starts a new solve but must not erase records from the
        # preceding environment in the same batched policy call.
        self.history = []

    def start_batch(self) -> None:
        return None

    def end_solve(self) -> None:
        return None

    def __call__(self, **state: Any) -> None:
        if int(state["step"]) != self.final_step:
            return
        costs = state["costs"][0].detach().float().cpu().numpy()
        candidates = state["candidates"][0].detach().float().cpu().numpy()
        mean = state["mean"][0].detach().float().cpu().numpy()
        order = np.argsort(costs, kind="stable")
        selected = order[: min(self.keep, len(order))]
        self.records.append(
            {
                "selected_first_action_normalized": mean[0, :5].tolist(),
                "elite_first_actions_normalized": candidates[selected, 0, :5].tolist(),
                "elite_costs": costs[selected].tolist(),
                "population_cost_mean": float(costs.mean()),
                "population_cost_std": float(costs.std()),
                "best_cost": float(costs[order[0]]),
                "best_candidate_first_action_normalized": candidates[order[0], 0, :5].tolist(),
            }
        )


def _position_array(infos: dict[str, Any]) -> np.ndarray:
    values = np.asarray(infos["privileged/block_0_pos"], dtype=np.float64)
    if values.ndim == 3:
        values = values[:, -1, :]
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"unexpected Cube block position shape: {values.shape}")
    return values


def _goal_positions(world: swm.World) -> np.ndarray:
    goals = []
    for env in world.envs.envs:
        raw = env.unwrapped
        target_block = int(getattr(raw, "_target_block", 0))
        goals.append(np.asarray(raw.cur_task_info["goal_xyzs"][target_block], dtype=np.float64))
    return np.asarray(goals, dtype=np.float64)


def _run_arm(
    *,
    policy_path: str,
    environment: str,
    device: str,
    seeds: list[int],
    action_mean: np.ndarray,
    action_std: np.ndarray,
    matrix: np.ndarray | None,
    bias: np.ndarray | None,
    horizon: int,
    receding_horizon: int,
    action_block: int,
    num_samples: int,
    n_steps: int,
    topk: int,
    var_scale: float,
    max_episode_steps: int,
    audit_steps: int,
    clip_env_actions: bool,
    reset_options: list[dict[str, Any]] | None = None,
    capture_initial_conditions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    model = load_compatible_pretrained(policy_path, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    trace = FinalCEMTrace(final_step=n_steps - 1, keep=min(topk, 8))
    planning_model: torch.nn.Module = model
    if matrix is not None and bias is not None:
        planning_model = ActionAdaptedCostModel(model, matrix, bias).to(device).eval()
    solver = swm.solver.CEMSolver(
        model=planning_model,
        batch_size=1,
        num_samples=num_samples,
        var_scale=var_scale,
        n_steps=n_steps,
        topk=topk,
        device=device,
        seed=seeds[0],
        callbacks=[trace],
    )
    action_normalizer = ActionNormalizer(
        action_mean,
        action_std,
        clip_output=clip_env_actions,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(
            horizon=horizon,
            receding_horizon=receding_horizon,
            history_len=3,
            action_block=action_block,
            warm_start=True,
        ),
        process={"action": action_normalizer},
        transform={"pixels": _image_transform(), "goal": _image_transform()},
    )
    world = swm.World(
        env_name=environment,
        num_envs=len(seeds),
        image_shape=(224, 224),
        max_episode_steps=max_episode_steps,
        goal_conditioned=True,
        pre_wrappers=[CubeGoalAliasWrapper],
        mode="task",
        ob_type="pixels",
        reward_task_id=2,
        terminate_at_goal=True,
        render_mode="rgb_array",
    )
    normalizer = action_normalizer
    try:
        world.set_policy(policy)
        # Reset explicitly so the evaluator can snapshot the initial state,
        # then run exactly one environment step without another reset.
        world.reset(
            seed=seeds[0],
            options=reset_options if reset_options is not None else {"render_goal": True},
        )
        if reset_options is not None and len(reset_options) != len(seeds):
            raise ValueError("strict paired reset options must match the number of seeds")
        if capture_initial_conditions is not None:
            for env in world.envs.envs:
                raw = env.unwrapped
                capture_initial_conditions.append(
                    {
                        "render_goal": True,
                        "variation_values": copy.deepcopy(raw.variation_space.value),
                        "state": np.concatenate(
                            [raw.data.qpos.copy(), raw.data.qvel.copy()]
                        ),
                    }
                )
        actual_seeds = np.asarray(world.envs.seeds, dtype=np.int64).copy()
        start_positions = _position_array(world.infos).copy()
        goals = _goal_positions(world)
        start_distances = np.linalg.norm(start_positions - goals, axis=1)

        distance_history: list[np.ndarray] = []

        def on_step(current_world: swm.World) -> None:
            # The values below are evaluator-only and are never inserted into
            # the policy info before the current action was selected.
            current_positions = _position_array(current_world.infos)
            current_distances = np.linalg.norm(current_positions - goals, axis=1)
            distance_history.append(current_distances.copy())

        world._run(max_steps=audit_steps, mode="wait", on_step=on_step)
        if len(distance_history) != audit_steps:
            raise RuntimeError(f"expected {audit_steps} audit steps, got {len(distance_history)}")
        after_distances = np.asarray(distance_history[-1], dtype=np.float64)
    finally:
        world.close()

    if len(trace.records) != len(seeds):
        raise RuntimeError(
            f"expected one final CEM trace per env, got {len(trace.records)} for {len(seeds)} envs"
        )
    transform_diagnostics = normalizer.diagnostics()
    records = []
    for index, trace_record in enumerate(trace.records):
        record = dict(trace_record)
        record["seed"] = int(actual_seeds[index])
        record["start_distance_m"] = float(start_distances[index])
        record["after_audit_steps_distance_m"] = float(after_distances[index])
        record["distance_change_after_audit_steps_m"] = float(after_distances[index] - start_distances[index])
        record["selected_first_action_raw"] = normalizer.inverse_transform(
            np.asarray(record["selected_first_action_normalized"], dtype=np.float32)
        ).tolist()
        records.append(record)
    return {
        "seeds": actual_seeds.tolist(),
        "records": records,
        "mean_start_distance_m": float(np.mean(start_distances)),
        "mean_after_audit_steps_distance_m": float(np.mean(after_distances)),
        "mean_distance_change_after_audit_steps_m": float(np.mean(after_distances - start_distances)),
        "improved_one_step_fraction": float(np.mean(after_distances < start_distances)),
        "action_transform_diagnostics": transform_diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/cube/lewm")
    parser.add_argument("--environment", default="swm/OGBCube-v0")
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261210)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--var-scale", type=float, default=1.0)
    parser.add_argument("--max-episode-steps", type=int, default=75)
    parser.add_argument(
        "--clip-env-actions",
        action="store_true",
        help="Clip inverse-transformed environment actions to [-1, 1].",
    )
    parser.add_argument(
        "--audit-steps",
        type=int,
        default=25,
        help="Evaluator-only rollout steps used to measure post-ranking progress.",
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for Cube ranking audit but is unavailable")
    if args.episodes < 2 or args.topk >= args.num_samples:
        raise ValueError("episodes must be >=2 and topk must be smaller than num-samples")
    if args.audit_steps < 1 or args.audit_steps > args.max_episode_steps:
        raise ValueError("audit-steps must be within the episode budget")
    matrix, bias, action_mean, action_std = _load_adapter(args.adapter_artifact, args.environment)
    seeds = [args.seed + i for i in range(args.episodes)]
    print("Auditing first-step Cube CEM ranking and evaluator-only physical progress on CUDA.")
    initial_conditions: list[dict[str, Any]] = []
    baseline = _run_arm(
        policy_path=args.policy, environment=args.environment, device=args.device,
        seeds=seeds, action_mean=action_mean, action_std=action_std,
        matrix=None, bias=None, horizon=args.horizon,
        receding_horizon=args.receding_horizon, action_block=args.action_block,
        num_samples=args.num_samples, n_steps=args.n_steps, topk=args.topk,
        var_scale=args.var_scale, max_episode_steps=args.max_episode_steps,
        audit_steps=args.audit_steps,
        clip_env_actions=args.clip_env_actions,
        capture_initial_conditions=initial_conditions,
    )
    adapted = _run_arm(
        policy_path=args.policy, environment=args.environment, device=args.device,
        seeds=seeds, action_mean=action_mean, action_std=action_std,
        matrix=matrix, bias=bias, horizon=args.horizon,
        receding_horizon=args.receding_horizon, action_block=args.action_block,
        num_samples=args.num_samples, n_steps=args.n_steps, topk=args.topk,
        var_scale=args.var_scale, max_episode_steps=args.max_episode_steps,
        audit_steps=args.audit_steps,
        clip_env_actions=args.clip_env_actions,
        reset_options=initial_conditions,
    )
    if baseline["seeds"] != adapted["seeds"]:
        raise RuntimeError("baseline and adapter seeds do not match")
    b = np.asarray([row["distance_change_after_audit_steps_m"] for row in baseline["records"]])
    a = np.asarray([row["distance_change_after_audit_steps_m"] for row in adapted["records"]])
    payload = {
        "schema": "live_cube_action_ranking_audit.v1",
        "protocol": {
            "checkpoint": args.policy,
            "checkpoint_frozen": True,
            "environment": args.environment,
            "device": args.device,
            "runtime_policy_state_access": False,
            "evaluator_privileged_state_role": "post_step_distance_only",
            "same_seed_pairing": True,
            "strict_initial_state_pairing": True,
            "strict_initial_state_source": "baseline qpos/qvel plus variation_values restored for adapter",
            "horizon": args.horizon,
            "receding_horizon": args.receding_horizon,
            "action_block": args.action_block,
            "cem": {"num_samples": args.num_samples, "n_steps": args.n_steps, "topk": args.topk},
            "cem_var_scale": args.var_scale,
            "max_episode_steps": args.max_episode_steps,
            "audit_steps": args.audit_steps,
            "adapter_artifact": str(args.adapter_artifact),
            "action_output_clip": args.clip_env_actions,
            "action_space_bounds": [-1.0, 1.0],
        },
        "baseline": baseline,
        "action_adapter": adapted,
        "paired": {
            "distance_change_adapter_minus_baseline_m": (a - b).tolist(),
            "adapter_more_progress_count": int(np.sum(a < b)),
            "baseline_more_progress_count": int(np.sum(a > b)),
            "ties": int(np.sum(a == b)),
            "mean_delta_m": float(np.mean(a - b)),
        },
        "claim_boundary": (
            "Evaluator-only first-step CEM ranking diagnostic for non-PushT Cube; "
            "not causal physics, not a planning success claim, and not a universal certificate."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ranking_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"baseline": baseline, "action_adapter": adapted, "paired": payload["paired"]}, indent=2))


if __name__ == "__main__":
    main()
