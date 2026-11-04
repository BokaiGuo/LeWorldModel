"""Execute old full-CEM prefix and re-score its tail after real RGB feedback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner
from experiments.spectral_reachability.reacher_pc_tail_carryover import cem_with_carried_tail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-cem", type=Path, required=True)
    parser.add_argument("--root-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    full, roots = (json.loads(p.read_text()) for p in (args.full_cem, args.root_source))
    if full["root_source_sha256"] != setup._sha256_file(args.root_source):
        raise RuntimeError("root source mismatch")
    policy, stats = Path(roots["protocol"]["policy"]), Path(roots["protocol"]["action_stats"])
    for key, path in (("weights_sha256", policy / "weights.pt"), ("action_stats_sha256", stats)):
        if setup._sha256_file(path) != full[key]:
            raise RuntimeError(f"{key} mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    setup.install_missing_field_compat()
    setup.validate_training_semantic_receipt(stats)
    mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
    _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
    model = setup.load_compatible_pretrained(policy, device="cuda").eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    before = setup._parameter_hash(model)
    if before != full["parameter_hash_before"]:
        raise RuntimeError("checkpoint mismatch")
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    rows = []
    try:
        setup.p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
        low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
        lower, upper = torch.as_tensor(low, device="cuda"), torch.as_tensor(high, device="cuda")
        for old in full["rows"]:
            seed = old["seed"]
            packets, index = setup.sample_history_packets(histories, seed=seed)
            root_row = next(r for r in roots["rows"] if r["seed"] == seed)
            context = setup.collect_executed_context(
                env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
            if (int(index) != root_row["history_index"] or
                    context["root_rgb_sha256"] != old["root_rgb_sha256"] or
                    context["goal_rgb_sha256"] != old["goal_rgb_sha256"]):
                raise RuntimeError("root identity mismatch")
            plan = torch.as_tensor(old["result"]["first_receipt"]["full_pre_feedback_plan_params"],
                                   dtype=torch.float32, device="cuda")
            if plan.shape != (1, 3, 10):
                raise ValueError("unexpected archived plan")

            def first(pixels, goal, history, root_seed):
                return plan[:, :1].clone(), {
                    "selection": "sealed original full-CEM plan",
                    "source_forward_rollout_evaluations": 192,
                    "source_predict_samples_horizon_weighted": 576}

            def carried_cem(model_arg, pixels, goal, history, **kw):
                if (kw["future_horizon"], kw["samples"], kw["steps"], kw["elite"], kw["seed"]) != (
                        2, 16, 2, 4, seed + 5000):
                    raise ValueError("unexpected suffix contract")

                def cost(candidates):
                    return setup.p0._cost_fn(model_arg, pixels, goal, history, candidates,
                                             action_lower=kw["action_lower"], action_upper=kw["action_upper"])

                return cem_with_carried_tail(cost, plan[:, 1:], samples=16, steps=2, elite=4,
                                             seed=kw["seed"])

            original_cem = runner.p0._cem
            try:
                runner.p0._cem = carried_cem
                result = runner._two_stage(
                    name="carryover", first_program=first, env=env, seed=seed, context=context,
                    preprocessor=processor, model=model, mean=mean, std=std, lower=lower, upper=upper,
                    action_block=5, progress_horizon=2, damping=0.0, device=torch.device("cuda"),
                    progress_solver="cem", cem_samples=16, cem_steps=2, cem_elite=4)
            finally:
                runner.p0._cem = original_cem
            if (result["first_action_witness_raw"] != old["result"]["first_action_witness_raw"] or
                    result["observed_successor_context"] != old["result"]["observed_successor_context"]):
                raise RuntimeError("first packet or observed successor does not reproduce source")
            rows.append({"seed": seed, "result": result})
            print(json.dumps({"seed": seed, "physical": result["physical"],
                              "suffix": result["suffix_receipt"]}), flush=True)
    finally:
        env.close()
    after = setup._parameter_hash(model)
    if before != after:
        raise RuntimeError("model parameters changed")
    report = {
        "scope": "known-root exploratory physical experiment; not a PC gate",
        "full_cem_sha256": setup._sha256_file(args.full_cem),
        "root_source_sha256": setup._sha256_file(args.root_source),
        "runner_sha256": setup._sha256_file(Path(__file__)),
        "parameter_hash_before": before, "parameter_hash_after": after,
        "cost_scope": "source initial search charged; source initial wall time not included",
        "rows": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
