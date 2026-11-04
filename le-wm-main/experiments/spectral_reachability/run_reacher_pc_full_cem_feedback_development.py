"""Known-root full-horizon CEM followed by actual-RGB suffix replanning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner


def first_packet_from_plan(plan, receipt):
    if plan.ndim != 3 or plan.shape[0] != 1 or plan.shape[1] != 3:
        raise ValueError("expected one three-packet CEM plan")
    return plan[:, :1].clone(), {
        **receipt,
        "full_pre_feedback_plan_params": plan.detach().cpu().tolist(),
        "executed_packet_index": 0,
        "planned_horizon_packets": 3,
        "predict_samples_horizon_weighted": int(receipt["forward_rollout_evaluations"]) * 3,
    }


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
    roots = json.loads(args.root_source.read_text())
    if setup._sha256_file(args.root_source) != source["source_report_sha256"]:
        raise RuntimeError("root source report mismatch")
    policy = Path(roots["protocol"]["policy"])
    stats = Path(roots["protocol"]["action_stats"])
    for key, path in (("weights_sha256", policy / "weights.pt"), ("action_stats_sha256", stats)):
        if setup._sha256_file(path) != source[key]:
            raise RuntimeError(f"source mismatch: {key}")
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
            packets, index = setup.sample_history_packets(histories, seed=seed)
            root_row = next(r for r in roots["rows"] if r["seed"] == seed)
            context = setup.collect_executed_context(
                env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
            if (int(index) != root_row["history_index"] or
                    context["root_rgb_sha256"] != row["root_rgb_sha256"] or
                    context["goal_rgb_sha256"] != row["goal_rgb_sha256"]):
                raise RuntimeError("root provenance mismatch")

            def first(pixels, goal, history, root_seed):
                plan, receipt = setup.p0._cem(
                    model, pixels, goal, history, future_horizon=3,
                    action_lower=lower, action_upper=upper,
                    samples=16, steps=12, elite=4, seed=root_seed + 9000)
                return first_packet_from_plan(plan, receipt)

            result = runner._two_stage(
                name="full_cem", first_program=first, env=env, seed=seed, context=context,
                preprocessor=processor, model=model, mean=mean, std=std, lower=lower, upper=upper,
                action_block=5, progress_horizon=2, damping=0.0, device=torch.device("cuda"),
                progress_solver="cem", cem_samples=16, cem_steps=2, cem_elite=4)
            result["suffix_receipt"]["predict_samples_horizon_weighted"] = (
                result["suffix_receipt"]["forward_rollout_evaluations"] * 2)
            rows.append({"seed": seed, "root_rgb_sha256": context["root_rgb_sha256"],
                         "goal_rgb_sha256": context["goal_rgb_sha256"], "result": result})
            print(json.dumps({"seed": seed, "physical": result["physical"]}), flush=True)
    finally:
        env.close()
    after = setup._parameter_hash(model)
    if before != after:
        raise RuntimeError("model parameters changed")
    report = {
        "scope": "known-root exploratory comparison, not independent confirmation or PC admission",
        "source_sha256": setup._sha256_file(args.source),
        "root_source_sha256": setup._sha256_file(args.root_source),
        "runner_sha256": setup._sha256_file(Path(__file__)),
        "weights_sha256": source["weights_sha256"], "action_stats_sha256": source["action_stats_sha256"],
        "parameter_hash_before": before, "parameter_hash_after": after,
        "cost_scope": "horizon-weighted nominal samples; full costs and equal-budget fairness uncertified",
        "rows": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
