"""Host-CUDA, model-only development smoke for predicted-successor CEM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import reacher_p1_predicted_successor_cem as selector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--action-stats", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20263160)
    parser.add_argument("--roots", type=int, default=2)
    parser.add_argument("--proposals", type=int, default=16)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--elite", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite development output")
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is required; CPU fallback is forbidden")
    if args.roots < 1:
        raise ValueError("roots must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    setup.install_missing_field_compat()
    setup.validate_training_semantic_receipt(args.action_stats)
    mean, std = setup._load_action_stats(args.action_stats, "swm/ReacherDMControl-v0")
    pool, histories, support, history_support = setup.load_runtime_support(
        args.action_stats, history_len=3, action_block=5,
    )
    model = setup.load_compatible_pretrained(args.policy, device="cuda").eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    before = setup._parameter_hash(model)
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65),
    )
    rows = []
    try:
        setup.p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
        low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
        lower = torch.as_tensor(low, device="cuda")
        upper = torch.as_tensor(high, device="cuda")
        for index in range(args.roots):
            seed = args.seed + index
            packets, history_index = setup.sample_history_packets(histories, seed=seed)
            context = setup.collect_executed_context(
                env, seed=seed, history_len=3, action_block=5,
                warmup_packets=packets, warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history",
            )
            bank, bank_receipt = setup.proposal_parameters(
                pool, proposals=args.proposals, horizon=1, seed=seed,
                mean=mean, std=std, lower=lower, upper=upper,
            )
            pixels, goal, history = setup.p0._prepare(
                processor, context["frames"], context["root_frame"], context["goal"],
                np.asarray(context["history_action_blocks"], dtype=np.float32), torch.device("cuda"),
            )
            params, receipt = selector.select_prefix_with_cem_suffix(
                model, pixels, goal, history, bank, future_horizon=2,
                action_lower=lower, action_upper=upper, samples=args.samples,
                steps=args.steps, elite=args.elite, seed=seed + 5000,
                non_greedy_margin=1e-6, future_margin=1e-6,
            )
            rows.append({
                "seed": seed, "history_index": int(history_index),
                "root_rgb_sha256": context["root_rgb_sha256"],
                "goal_rgb_sha256": context["goal_rgb_sha256"],
                "bank_receipt": bank_receipt,
                "selected_params": None if params is None else params.cpu().tolist(),
                "selection": receipt,
            })
            print(json.dumps({"seed": seed, "selected": receipt["selected_proposal_index"]}), flush=True)
    finally:
        env.close()
    after = setup._parameter_hash(model)
    report = {
        "scope": "model-only development smoke; no P1 admission or physical candidate execution",
        "cuda_device": torch.cuda.get_device_name(),
        "inputs": {"weights_sha256": setup._sha256_file(args.policy / "weights.pt"),
                   "action_stats_sha256": setup._sha256_file(args.action_stats),
                   "selector_sha256": setup._sha256_file(Path(selector.__file__))},
        "parameter_hash_before": before, "parameter_hash_after": after,
        "frozen_parameters_unchanged": before == after,
        "planning_input": "RGB, goal RGB and actual action history; no learned qpos readout",
        "protocol": vars(args) | {"output": str(args.output), "policy": str(args.policy),
                                  "action_stats": str(args.action_stats)},
        "source_support": support, "history_support": history_support, "rows": rows,
    }
    if before != after:
        raise RuntimeError("frozen model parameters changed")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
