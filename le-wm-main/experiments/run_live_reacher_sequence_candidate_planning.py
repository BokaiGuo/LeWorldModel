"""Live RGB Reacher planning with a fixed pulse/recovery candidate bank.

This is the closed-loop consequence of the sequence-bank ranking audit.  The
LeWM checkpoint, encoder, and predictor remain frozen.  The bank is injected
into CEM under the same sample budget and is scored from rendered RGB only.
"""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path

if __package__ in (None, ""):
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import torch
# Keep CUDA initialization ahead of the headless MuJoCo backend setup.  The
# managed launcher can make PyTorch's device probe fail when EGL is exported
# before torch is imported, so configure EGL inside the process after torch is
# loaded.
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm

from experiments.audit_live_reacher_shared_candidate_ranking import _candidate_bank
from experiments.audit_reacher_sequence_candidate_ranking import _sequence_bank
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.capacity_readout_runtime import (
    CapacityReadoutCostModel,
    apply_capacity_bilinear_mode,
    load_capacity_readout,
)
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import (
    ReacherGoalImageWrapper,
    _image_transform,
    _load_action_stats,
    install_missing_field_compat,
)
from experiments.run_live_tworoom_planning import CandidateBankCEMSolver


def validate_reacher_capacity_contract(
    artifact: dict[str, object],
) -> None:
    """Reject readouts trained for a different task/phase contract."""
    expected = {
        "target_mode": "absolute",
        "goal_conditioning": "none",
        "phase_conditioning": "none",
        "action_frame": {"raw", "proposal_relative"},
    }
    mismatches = {
        key: (artifact.get(key), value)
        for key, value in expected.items()
        if (
            artifact.get(key, "none") not in value
            if isinstance(value, set)
            else artifact.get(key, "none") != value
        )
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {sorted(wanted)!r})"
            if isinstance(wanted, set)
            else f"{key}={actual!r} (expected {wanted!r})"
            for key, (actual, wanted) in mismatches.items()
        )
        raise ValueError(
            "capacity readout violates the Reacher absolute-qpos contract: "
            + details
        )


def _planning_policy(args, action_mean: np.ndarray, action_std: np.ndarray):
    model = load_compatible_pretrained(args.policy, device=args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    if args.bank_family == "capacity17":
        raw_bank, candidate_bank = _candidate_bank(
            action_mean, action_std, horizon=args.horizon, action_block=args.action_block,
        )
        labels = [
            {"family": "constant", "direction": np.asarray(direction).tolist()}
            for direction in raw_bank
        ]
    else:
        _, candidate_bank, labels = _sequence_bank(
            action_mean,
            action_std,
            horizon=args.horizon,
            action_block=args.action_block,
            bank_family=args.bank_family,
        )
    if args.cost_metric == "capacity_safe_readout":
        if args.capacity_readout_artifact is None:
            raise ValueError(
                "--capacity-readout-artifact is required for capacity_safe_readout"
            )
        artifact = load_capacity_readout(Path(args.capacity_readout_artifact))
        validate_reacher_capacity_contract(artifact)
        artifact = apply_capacity_bilinear_mode(
            artifact, args.capacity_bilinear_mode,
        )
        if args.capacity_bilinear_base_cost_blend_weight is not None:
            artifact["bilinear_base_cost_blend_weight"] = float(
                args.capacity_bilinear_base_cost_blend_weight
            )
        if args.capacity_bilinear_start_iteration is not None:
            artifact["bilinear_start_iteration"] = int(
                args.capacity_bilinear_start_iteration
            )
        if args.capacity_correction_multiplier is not None:
            if not 0.0 <= args.capacity_correction_multiplier <= 1.0:
                raise ValueError("capacity correction multiplier must be in [0, 1]")
            artifact["correction_multiplier"] = float(args.capacity_correction_multiplier)
        if args.capacity_base_rank_quantile is not None:
            if not 0.0 < args.capacity_base_rank_quantile <= 1.0:
                raise ValueError("capacity base rank quantile must be in (0, 1]")
            artifact["base_rank_quantile"] = float(args.capacity_base_rank_quantile)
        if args.capacity_base_cost_correction:
            artifact["base_cost_correction"] = True
        if args.capacity_base_cost_blend_weight is not None:
            if not 0.0 <= args.capacity_base_cost_blend_weight <= 1.0:
                raise ValueError("capacity base-cost blend weight must be in [0, 1]")
            artifact["mlp_base_cost_blend_weight"] = float(
                args.capacity_base_cost_blend_weight
            )
        if args.capacity_shuffle_base_cost_control:
            artifact["mlp_shuffle_base_cost_control"] = True
        if args.capacity_base_cost_elite_k is not None:
            if args.capacity_base_cost_elite_k < 1:
                raise ValueError("capacity base-cost elite k must be positive")
            artifact["mlp_base_cost_elite_k"] = int(
                args.capacity_base_cost_elite_k
            )
        model = CapacityReadoutCostModel(model, artifact, action_mean, action_std)
    if args.num_samples <= int(candidate_bank.shape[0]) + 1:
        raise ValueError(
            "num-samples must exceed sequence candidate count plus the mean slot: "
            f"{args.num_samples} <= {int(candidate_bank.shape[0]) + 1}"
        )
    solver = CandidateBankCEMSolver(
        model=model,
        batch_size=1,
        num_samples=args.num_samples,
        var_scale=1.0,
        n_steps=args.n_steps,
        topk=args.topk,
        device=args.device,
        seed=args.seed,
        candidate_bank=candidate_bank,
        candidate_bank_iterations=args.candidate_bank_iterations,
        final_action_selection=args.final_action_selection,
    )
    return (
        swm.policy.WorldModelPolicy(
            solver=solver,
            config=swm.PlanConfig(
                horizon=args.horizon,
                receding_horizon=args.receding_horizon,
                history_len=3,
                action_block=args.action_block,
                warm_start=True,
            ),
            process={"action": ActionNormalizer(action_mean, action_std)},
            transform={"pixels": _image_transform(), "goal": _image_transform()},
        ),
        labels,
    )


def _qpos_distance(current: np.ndarray, target: np.ndarray) -> float:
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if current.size < 2 or target.size < 2:
        raise ValueError("Reacher qpos distance requires at least two joints")
    delta = (current[:2] - target[:2] + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.linalg.norm(delta))


def _world_qpos_distance(world: swm.World, env_index: int) -> float:
    env = world.envs.envs[int(env_index)].unwrapped
    current = np.asarray(env.env.physics.data.qpos, dtype=np.float64)
    target = np.asarray(env.env.task.target_qpos, dtype=np.float64)
    return _qpos_distance(current, target)


def _evaluate_with_qpos_metrics(world: swm.World, episodes: int, seed: int) -> dict:
    """Evaluate while recording privileged metrics, never policy inputs."""
    successes = np.zeros(episodes, dtype=np.int64)
    seeds = np.zeros(episodes, dtype=np.int64)
    final_distance = np.full(episodes, np.nan, dtype=np.float64)
    min_by_env = np.full(world.num_envs, np.inf, dtype=np.float64)
    min_distance = np.full(episodes, np.nan, dtype=np.float64)

    def on_step(active_world):
        for env_index in range(active_world.num_envs):
            min_by_env[env_index] = min(
                min_by_env[env_index],
                _world_qpos_distance(active_world, env_index),
            )

    def on_done(env_index, episode_index, active_world):
        successes[episode_index] = int(active_world.terminateds[env_index])
        seeds[episode_index] = int(active_world.envs.seeds[env_index])
        final_distance[episode_index] = _world_qpos_distance(active_world, env_index)
        min_distance[episode_index] = min_by_env[env_index]

    world._run(
        episodes=episodes,
        seed=seed,
        mode="wait",
        on_step=on_step,
        on_done=on_done,
    )
    if np.any(~np.isfinite(final_distance)) or np.any(~np.isfinite(min_distance)):
        raise RuntimeError("Reacher evaluator did not record every episode distance")
    return {
        "episode_successes": successes,
        "seeds": seeds,
        "final_qpos_distance": final_distance,
        "min_qpos_distance": min_distance,
    }


def _run(args, kind: str, action_mean: np.ndarray, action_std: np.ndarray):
    if kind == "random":
        policy = swm.policy.RandomPolicy(seed=args.seed)
        labels = None
    else:
        policy, labels = _planning_policy(args, action_mean, action_std)
    world = swm.World(
        env_name="swm/ReacherDMControl-v0",
        num_envs=args.episodes,
        image_shape=(224, 224),
        max_episode_steps=args.max_episode_steps,
        goal_conditioned=True,
        pre_wrappers=[partial(ReacherGoalImageWrapper, target_delta=args.target_delta)],
        render_mode="rgb_array",
        task="qpos_match",
    )
    try:
        world.set_policy(policy)
        result = _evaluate_with_qpos_metrics(world, args.episodes, args.seed)
    finally:
        world.close()
    successes = np.asarray(result["episode_successes"], dtype=np.int64)
    output = {
        "success_rate": float(successes.mean()),
        "episode_successes": successes.tolist(),
        "seeds": np.asarray(result["seeds"], dtype=np.int64).tolist(),
        "final_qpos_distance": np.asarray(
            result["final_qpos_distance"], dtype=np.float64,
        ).tolist(),
        "min_qpos_distance": np.asarray(
            result["min_qpos_distance"], dtype=np.float64,
        ).tolist(),
        "mean_final_qpos_distance": float(np.mean(result["final_qpos_distance"])),
        "mean_min_qpos_distance": float(np.mean(result["min_qpos_distance"])),
        "candidate_labels": labels,
    }
    if kind != "random" and isinstance(policy.solver.model, CapacityReadoutCostModel):
        output["runtime_diagnostics"] = policy.solver.model.runtime_diagnostics()
    return output


def _canonicalize(result: dict) -> dict:
    order = np.argsort(np.asarray(result["seeds"], dtype=np.int64))
    output = dict(result)
    output["seeds"] = np.asarray(result["seeds"], dtype=np.int64)[order].tolist()
    output["episode_successes"] = np.asarray(result["episode_successes"], dtype=np.int64)[order].tolist()
    for key in ("final_qpos_distance", "min_qpos_distance"):
        output[key] = np.asarray(result[key], dtype=np.float64)[order].tolist()
    return output


def main() -> None:
    install_missing_field_compat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm")
    parser.add_argument(
        "--adapter-artifact", type=Path,
        default=Path("results/live_cross_environment_prediction_20261015/reacher_cuda/cross_environment_prediction.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261600)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--bank-family", choices=("full", "compact", "support4", "capacity17"), default="full",
        help="fixed support bank; support4 injects only four midpoint pulse-recovery candidates",
    )
    parser.add_argument(
        "--candidate-bank-iterations", type=int, default=None,
        help="if set, inject the fixed bank only for this many CEM iterations",
    )
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument(
        "--final-action-selection", choices=("mean", "best"), default="mean",
        help="execute the final elite mean or the final top-1 candidate",
    )
    parser.add_argument(
        "--cost-metric", choices=("euclidean", "capacity_safe_readout"), default="euclidean",
    )
    parser.add_argument("--capacity-readout-artifact", type=Path, default=None)
    parser.add_argument(
        "--capacity-bilinear-mode",
        choices=(
            "artifact",
            "action_only",
            "full_predicted_dynamics",
            "consensus",
            "rank_fusion",
        ),
        default="artifact",
        help=(
            "frozen exported bilinear component; use artifact for the train-only "
            "selection, or an explicit component ablation"
        ),
    )
    parser.add_argument(
        "--capacity-bilinear-base-cost-blend-weight",
        type=float,
        default=None,
        help=(
            "fixed conservative blend of frozen base cost into the bilinear "
            "bank ranking; omitted means the artifact's declared value"
        ),
    )
    parser.add_argument(
        "--capacity-bilinear-start-iteration",
        type=int,
        default=None,
        help=(
            "fixed CEM iteration at which bilinear bank ranking starts; "
            "use with candidate-bank-iterations covering the final stage"
        ),
    )
    parser.add_argument(
        "--capacity-correction-multiplier", type=float, default=None,
        help="optional predeclared conservative override for a capacity readout",
    )
    parser.add_argument(
        "--capacity-base-rank-quantile", type=float, default=None,
        help="optional predeclared base-cost rank gate for a capacity readout",
    )
    parser.add_argument(
        "--capacity-base-cost-correction", action="store_true",
        help="keep frozen model cost and apply the readout only as an additive correction",
    )
    parser.add_argument(
        "--capacity-base-cost-blend-weight",
        type=float,
        default=None,
        help=(
            "fixed weight on frozen LeWM cost after rank-affine source-action repair; "
            "zero recovers the source-action-only controller"
        ),
    )
    parser.add_argument(
        "--capacity-shuffle-base-cost-control",
        action="store_true",
        help="roll frozen LeWM costs within each proposal cloud as a dynamics-placebo control",
    )
    parser.add_argument(
        "--capacity-base-cost-elite-k",
        type=int,
        default=None,
        help=(
            "allow frozen LeWM cost to tie-break only within the top-k "
            "source-action candidates; one disables the tie-break"
        ),
    )
    parser.add_argument("--target-delta", type=float, nargs=2, default=(0.65, -0.65))
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.episodes < 2 or args.topk >= args.num_samples:
        raise ValueError("require episodes >= 2 and topk < num-samples")
    environment = "swm/ReacherDMControl-v0"
    action_mean, action_std = _load_action_stats(args.adapter_artifact, environment)
    print("Running live RGB Reacher sequence-bank CEM versus random, same seeds.")
    random_result = _canonicalize(_run(args, "random", action_mean, action_std))
    planning_result = _canonicalize(_run(args, "planning", action_mean, action_std))
    if random_result["seeds"] != planning_result["seeds"]:
        raise RuntimeError("random and planning did not evaluate the same seed set")
    random_success = np.asarray(random_result["episode_successes"], dtype=np.int64)
    planning_success = np.asarray(planning_result["episode_successes"], dtype=np.int64)
    paired_delta = planning_success - random_success
    payload = {
        "schema": "live_reacher_sequence_candidate_planning.v1",
        "protocol": {
            "checkpoint": args.policy,
            "checkpoint_frozen": True,
            "runtime_input": "rendered RGB current frame, rendered RGB target-qpos frame, and action/history only",
            "runtime_state_access": False,
            "privileged_state_used_for_evaluation_metrics_only": True,
            "same_seed_pairing": True,
            "target_delta": [float(value) for value in args.target_delta],
            "device": args.device,
            "candidate_bank": args.bank_family,
            "candidate_count": {
                "support4": 4,
                "compact": 49,
                "full": 161,
                "capacity17": 17,
            }[args.bank_family],
            "cost_metric": args.cost_metric,
            "capacity_readout_artifact": (
                str(args.capacity_readout_artifact)
                if args.capacity_readout_artifact else None
            ),
            "capacity_bilinear_mode": args.capacity_bilinear_mode,
            "capacity_bilinear_base_cost_blend_weight": (
                args.capacity_bilinear_base_cost_blend_weight
            ),
            "capacity_bilinear_start_iteration": args.capacity_bilinear_start_iteration,
            "capacity_correction_multiplier_override": args.capacity_correction_multiplier,
            "capacity_base_rank_quantile_override": args.capacity_base_rank_quantile,
            "capacity_base_cost_correction": args.capacity_base_cost_correction,
            "capacity_base_cost_blend_weight": args.capacity_base_cost_blend_weight,
            "capacity_shuffle_base_cost_control": args.capacity_shuffle_base_cost_control,
            "capacity_base_cost_elite_k": args.capacity_base_cost_elite_k,
            "candidate_bank_in_cem_budget": True,
            "candidate_bank_iterations": args.candidate_bank_iterations,
            "cem": {"num_samples": args.num_samples, "n_steps": args.n_steps, "topk": args.topk},
            "final_action_selection": args.final_action_selection,
            "horizon": args.horizon,
            "receding_horizon": args.receding_horizon,
            "action_block": args.action_block,
        },
        "random": random_result,
        "frozen_lewm": planning_result,
        "paired": {
            "seed_order": random_result["seeds"],
            "delta_lewm_minus_random": paired_delta.tolist(),
            "wins": int(np.sum(paired_delta > 0)),
            "losses": int(np.sum(paired_delta < 0)),
            "ties": int(np.sum(paired_delta == 0)),
            "success_rate_delta": float(paired_delta.mean()),
        },
        "claim_boundary": "Independent live Reacher RGB sequence-bank consequence; no universal world-model or causal certificate claim.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "planning.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"random": random_result, "frozen_lewm": planning_result}, indent=2))


if __name__ == "__main__":
    main()
