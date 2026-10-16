"""Paired CUDA/EGL Cube planning with a frozen phase-conditioned candidate bank.

The baseline is the released frozen RGB CEM planner.  The intervention uses
the same frozen checkpoint and CEM budget, but initializes each solve from a
four-way generic action-support bank (x+/x-/y+/y-) containing lateral
alignment, descent, gripper closure, and a short push.  The bank is ranked by
the frozen model cost and never uses privileged cube state or planning-test
outcomes.  This is an action-support intervention, not a claim of universal
world-model transfer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.run_live_cube_action_adapter_planning import (
    _canonicalize,
    _load_adapter,
    _run_controller,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/cube/lewm")
    parser.add_argument("--environment", default="swm/OGBCube-v0")
    parser.add_argument(
        "--adapter-artifact",
        type=Path,
        default=Path("results/live_cross_environment_prediction_20261015/cube_action_adapter_cuda/cross_environment_prediction.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261045)
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
        help="Clip inverse-transformed actions to the declared Cube [-1,1] environment bounds.",
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for phase-conditioned Cube planning but unavailable")
    if args.episodes < 2 or args.topk >= args.num_samples:
        raise ValueError("episodes must be >=2 and topk must be smaller than num-samples")
    if args.horizon < 5:
        raise ValueError("phase-conditioned Cube planning requires horizon >= 5")
    if args.max_episode_steps < args.horizon * args.action_block:
        raise ValueError("max-episode-steps must cover one planned horizon")

    _, _, action_mean, action_std = _load_adapter(args.adapter_artifact, args.environment)
    seeds = [args.seed + i for i in range(args.episodes)]
    print("Running CUDA phase-conditioned Cube planning: frozen CEM versus frozen CEM with model-ranked action-support bank.")
    initial_conditions: list[dict] = []
    baseline = _canonicalize(_run_controller(
        policy_path=args.policy,
        environment=args.environment,
        device=args.device,
        seeds=seeds,
        action_mean=action_mean,
        action_std=action_std,
        matrix=None,
        bias=None,
        horizon=args.horizon,
        receding_horizon=args.receding_horizon,
        action_block=args.action_block,
        num_samples=args.num_samples,
        n_steps=args.n_steps,
        topk=args.topk,
        var_scale=args.var_scale,
        max_episode_steps=args.max_episode_steps,
        clip_env_actions=args.clip_env_actions,
        phase_conditioned=False,
        capture_initial_conditions=initial_conditions,
    ))
    phase = _canonicalize(_run_controller(
        policy_path=args.policy,
        environment=args.environment,
        device=args.device,
        seeds=seeds,
        action_mean=action_mean,
        action_std=action_std,
        matrix=None,
        bias=None,
        horizon=args.horizon,
        receding_horizon=args.receding_horizon,
        action_block=args.action_block,
        num_samples=args.num_samples,
        n_steps=args.n_steps,
        topk=args.topk,
        var_scale=args.var_scale,
        max_episode_steps=args.max_episode_steps,
        clip_env_actions=args.clip_env_actions,
        phase_conditioned=True,
        reset_options=initial_conditions,
    ))
    if baseline["seeds"] != phase["seeds"]:
        raise RuntimeError("baseline and phase-conditioned arms did not use the same seeds")
    baseline_success = np.asarray(baseline["episode_successes"], dtype=np.int64)
    phase_success = np.asarray(phase["episode_successes"], dtype=np.int64)
    baseline_distance = np.asarray(baseline["evaluator_metrics"]["final_block_goal_distance_m"], dtype=np.float64)
    phase_distance = np.asarray(phase["evaluator_metrics"]["final_block_goal_distance_m"], dtype=np.float64)
    success_delta = phase_success - baseline_success
    distance_delta = phase_distance - baseline_distance
    payload = {
        "schema": "live_cube_phase_conditioned_planning.v1",
        "protocol": {
            "checkpoint": args.policy,
            "checkpoint_frozen": True,
            "environment": args.environment,
            "device": args.device,
            "runtime_input": "live RGB current frame, rendered RGB goal, and action/history only",
            "runtime_state_access": False,
            "same_seed_pairing": True,
            "strict_initial_state_pairing": True,
            "phase_bank": "four generic x/y directions; align, descend, close, push; model-cost ranked",
            "planning_outcomes_used_for_bank": False,
            "horizon": args.horizon,
            "receding_horizon": args.receding_horizon,
            "action_block": args.action_block,
            "cem": {"num_samples": args.num_samples, "n_steps": args.n_steps, "topk": args.topk},
            "cem_var_scale": args.var_scale,
            "max_episode_steps": args.max_episode_steps,
            "action_output_clip": args.clip_env_actions,
        },
        "baseline": baseline,
        "phase_conditioned": phase,
        "paired": {
            "seed_order": baseline["seeds"],
            "success_delta_phase_minus_baseline": success_delta.tolist(),
            "success_wins": int(np.sum(success_delta > 0)),
            "success_losses": int(np.sum(success_delta < 0)),
            "success_ties": int(np.sum(success_delta == 0)),
            "success_rate_delta": float(success_delta.mean()),
            "final_distance_delta_phase_minus_baseline_m": distance_delta.tolist(),
            "distance_wins": int(np.sum(distance_delta < 0.0)),
            "distance_losses": int(np.sum(distance_delta > 0.0)),
            "distance_ties": int(np.sum(distance_delta == 0.0)),
            "mean_final_distance_delta_m": float(distance_delta.mean()),
        },
        "claim_boundary": "Independent Cube action-support intervention; not a universal world-model, causal-physics, or commutator certificate.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "planning.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["paired"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
