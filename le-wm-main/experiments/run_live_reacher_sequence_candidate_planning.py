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
from experiments.capacity_readout_runtime import CapacityReadoutCostModel, load_capacity_readout
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import (
    ReacherGoalImageWrapper,
    _image_transform,
    _load_action_stats,
    install_missing_field_compat,
)
from experiments.run_live_tworoom_planning import CandidateBankCEMSolver


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
        result = world.evaluate(episodes=args.episodes, seed=args.seed, reset_mode="wait")
    finally:
        world.close()
    successes = np.asarray(result["episode_successes"], dtype=np.int64)
    return {
        "success_rate": float(successes.mean()),
        "episode_successes": successes.tolist(),
        "seeds": np.asarray(result["seeds"], dtype=np.int64).tolist(),
        "candidate_labels": labels,
    }


def _canonicalize(result: dict) -> dict:
    order = np.argsort(np.asarray(result["seeds"], dtype=np.int64))
    output = dict(result)
    output["seeds"] = np.asarray(result["seeds"], dtype=np.int64)[order].tolist()
    output["episode_successes"] = np.asarray(result["episode_successes"], dtype=np.int64)[order].tolist()
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
        "--cost-metric", choices=("euclidean", "capacity_safe_readout"), default="euclidean",
    )
    parser.add_argument("--capacity-readout-artifact", type=Path, default=None)
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
            "capacity_correction_multiplier_override": args.capacity_correction_multiplier,
            "capacity_base_rank_quantile_override": args.capacity_base_rank_quantile,
            "capacity_base_cost_correction": args.capacity_base_cost_correction,
            "candidate_bank_in_cem_budget": True,
            "candidate_bank_iterations": args.candidate_bank_iterations,
            "cem": {"num_samples": args.num_samples, "n_steps": args.n_steps, "topk": args.topk},
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
