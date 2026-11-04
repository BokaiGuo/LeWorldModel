"""Real-feedback physical development comparison of a sealed prefix bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner
from experiments.spectral_reachability.run_reacher_successor_cem_physical_development import bound_indices


def physical_indices(row, bank):
    """Keep three fixed controls and every appended proposal, without outcomes."""
    base = int(row["original_bank_size"])
    if not 2 <= base <= len(bank) or len(row["refinement"]["backtracking"]) != 4:
        raise ValueError("invalid sealed refinement bank")
    controls = bound_indices(row, bank)
    appended = [r for r in row["refinement"]["backtracking"] if r["appended"]]
    if len(appended) != len(bank) - base:
        raise ValueError("appended candidates disagree with backtracking receipt")
    controls.update({f"refined_{i}": base + i for i in range(len(appended))})
    return controls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite output")
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    source = json.loads(args.source.read_text())
    root = json.loads(args.root_source.read_text())
    if setup._sha256_file(args.root_source) != source["source_report_sha256"]:
        raise RuntimeError("root source report mismatch")
    protocol = source["protocol"]
    policy, stats = Path(root["protocol"]["policy"]), Path(root["protocol"]["action_stats"])
    for key, path in (("weights_sha256", policy / "weights.pt"), ("action_stats_sha256", stats)):
        if setup._sha256_file(path) != source[key]:
            raise RuntimeError(f"source mismatch: {key}")
    if source["parameter_hash_before"] != source["parameter_hash_after"]:
        raise RuntimeError("source model was not frozen")
    args.output.mkdir(parents=True, exist_ok=False)
    setup.install_missing_field_compat()
    setup.validate_training_semantic_receipt(stats)
    mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
    _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
    model = setup.load_compatible_pretrained(policy, device="cuda").eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    before = setup._parameter_hash(model)
    if before != source["parameter_hash_before"]:
        raise RuntimeError("model differs from source")
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
                raise RuntimeError("root, history or goal mismatch")
            bank = torch.as_tensor(row["bank_params"], dtype=torch.float32, device="cuda")
            indices = physical_indices(row, bank)
            methods = {}
            for name, index in indices.items():
                def first(pixels, goal, history, root_seed, *, candidate=index):
                    return bank[candidate:candidate + 1], {
                        "proposal_index": candidate, "selection": "sealed model-only report"}
                methods[name] = runner._two_stage(
                    name=name, first_program=first, env=env, seed=seed, context=context,
                    preprocessor=processor, model=model, mean=mean, std=std, lower=lower, upper=upper,
                    action_block=5, progress_horizon=2, damping=0.0, device=torch.device("cuda"),
                    progress_solver="cem", cem_samples=protocol["samples"],
                    cem_steps=protocol["steps"], cem_elite=protocol["elite"])
                print(json.dumps({"seed": seed, "arm": name, "physical": methods[name]["physical"]}), flush=True)
            rows.append({"seed": seed, "indices": indices, "methods": methods})
    finally:
        env.close()
    after = setup._parameter_hash(model)
    if before != after:
        raise RuntimeError("model parameters changed")
    report = {"scope": "known-root candidate diagnosis; not independent or P1 admission",
              "input_report_sha256": setup._sha256_file(args.source),
              "root_source_sha256": setup._sha256_file(args.root_source),
              "runner_sha256": setup._sha256_file(Path(__file__)),
              "parameter_hash_before": before, "parameter_hash_after": after,
              "feedback": "each arm replans from its own real RGB and actual action history",
              "cost_scope": "source selection and execution costs separated; full cost not certified",
              "rows": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
