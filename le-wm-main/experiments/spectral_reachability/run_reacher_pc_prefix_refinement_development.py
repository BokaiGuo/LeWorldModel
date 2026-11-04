"""Run nonlinear spectral refinement on known Reacher roots with CUDA."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import reacher_p1_predicted_successor_cem as selector
from experiments.spectral_reachability import reacher_pc_guarded_spectral as compiler
from experiments.spectral_reachability import reacher_pc_prefix_refinement as refinement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite output")
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    source = json.loads(args.source.read_text())
    protocol = source["protocol"]
    policy, stats = Path(protocol["policy"]), Path(protocol["action_stats"])
    for key, path in (("weights_sha256", policy / "weights.pt"), ("action_stats_sha256", stats)):
        if setup._sha256_file(path) != source["inputs"][key]:
            raise RuntimeError(f"source mismatch: {key}")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    setup.install_missing_field_compat()
    setup.validate_training_semantic_receipt(stats)
    mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
    _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
    model = setup.load_compatible_pretrained(policy, device="cuda").eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    before = setup._parameter_hash(model)
    if before != source["parameter_hash_before"]:
        raise RuntimeError("model does not match source")
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    rows = []
    try:
        setup.p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
        low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
        lower, upper = torch.as_tensor(low, device="cuda"), torch.as_tensor(high, device="cuda")
        for row in source["rows"]:
            seed = row["seed"]
            packets, history_index = setup.sample_history_packets(histories, seed=seed)
            context = setup.collect_executed_context(
                env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
            if (int(history_index) != row["history_index"]
                    or context["root_rgb_sha256"] != row["root_rgb_sha256"]
                    or context["goal_rgb_sha256"] != row["goal_rgb_sha256"]):
                raise RuntimeError("root provenance mismatch")
            pixels, goal, history = setup.p0._prepare(
                processor, context["frames"], context["root_frame"], context["goal"],
                np.asarray(context["history_action_blocks"], dtype=np.float32), torch.device("cuda"))
            bank, geometry = compiler.compile_reacher(model, pixels, goal, history, lower=lower, upper=upper)
            augmented, refining = refinement.augment(model, pixels, goal, history, bank, lower=lower, upper=upper)
            chosen, receipt = selector.select_prefix_with_cem_suffix(
                model, pixels, goal, history, augmented, future_horizon=2, action_lower=lower,
                action_upper=upper, samples=protocol["samples"], steps=protocol["steps"],
                elite=protocol["elite"], seed=seed + 5000, non_greedy_margin=1e-6, future_margin=1e-6)
            rows.append({"seed": seed, "history_index": int(history_index),
                         "root_rgb_sha256": context["root_rgb_sha256"],
                         "goal_rgb_sha256": context["goal_rgb_sha256"], "geometry": geometry,
                         "original_bank_size": len(bank), "bank_params": augmented.cpu().tolist(),
                         "refinement": refining, "selection": receipt,
                         "selected_params": None if chosen is None else chosen.cpu().tolist()})
            print(json.dumps({"seed": seed, "original_count": len(bank), "augmented_count": len(augmented),
                              "selected": receipt["selected_proposal_index"]}), flush=True)
    finally:
        env.close()
    after = setup._parameter_hash(model)
    if before != after:
        raise RuntimeError("frozen parameters changed")
    report = {"scope": "known-root model-only development; not P1 admission",
              "source_report_sha256": setup._sha256_file(args.source),
              "weights_sha256": source["inputs"]["weights_sha256"],
              "action_stats_sha256": source["inputs"]["action_stats_sha256"],
              "compiler_sha256": setup._sha256_file(Path(compiler.__file__)),
              "refinement_sha256": setup._sha256_file(Path(refinement.__file__)),
              "selector_sha256": setup._sha256_file(Path(selector.__file__)),
              "parameter_hash_before": before, "parameter_hash_after": after,
              "protocol": {"iterations": 4, "lr": 0.08, "radius": 0.25,
                           "temperature": 0.01, "penalty": 10.0,
                           "samples": protocol["samples"], "steps": protocol["steps"],
                           "elite": protocol["elite"], "future_horizon": 2},
              "wall_seconds_including_initialization": time.perf_counter() - started,
              "cost_scope": "separate compiler, refinement, and selector; encoder costs uncertified",
              "rows": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
