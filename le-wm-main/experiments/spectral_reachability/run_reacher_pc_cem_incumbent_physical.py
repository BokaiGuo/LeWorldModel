"""Execute sealed CEM incumbent prefixes with actual-RGB suffix planning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--full-cem", type=Path, required=True)
    parser.add_argument("--root-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    audit, full, roots = (json.loads(p.read_text()) for p in (args.audit, args.full_cem, args.root_source))
    if (audit["full_cem_sha256"] != setup._sha256_file(args.full_cem) or
            audit["root_source_sha256"] != setup._sha256_file(args.root_source)):
        raise RuntimeError("report provenance mismatch")
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
    if before != audit["parameter_hash_before"]:
        raise RuntimeError("checkpoint parameter mismatch")
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    rows = []
    try:
        setup.p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
        low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
        lower, upper = torch.as_tensor(low, device="cuda"), torch.as_tensor(high, device="cuda")
        for row in audit["rows"]:
            seed = row["seed"]
            packets, index = setup.sample_history_packets(histories, seed=seed)
            root_row = next(r for r in roots["rows"] if r["seed"] == seed)
            context = setup.collect_executed_context(
                env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
            if (int(index) != root_row["history_index"] or
                    context["root_rgb_sha256"] != row["root_rgb_sha256"] or
                    context["goal_rgb_sha256"] != row["goal_rgb_sha256"]):
                raise RuntimeError("root identity mismatch")
            plan = torch.as_tensor(row["best_plan_params"], dtype=torch.float32, device="cuda")
            if plan.shape != (1, 3, 10):
                raise ValueError("unexpected incumbent plan")

            def first(pixels, goal, history, root_seed):
                return plan[:, :1].clone(), {
                    "selection": "sealed incumbent audit", "best_iteration": row["best_iteration"],
                    "source_forward_rollout_evaluations": row["forward_rollout_evaluations"],
                    "source_predict_samples_horizon_weighted": row["predict_samples_horizon_weighted"],
                    "original_full_plan_params": row["best_plan_params"]}

            result = runner._two_stage(
                name="cem_incumbent", first_program=first, env=env, seed=seed, context=context,
                preprocessor=processor, model=model, mean=mean, std=std, lower=lower, upper=upper,
                action_block=5, progress_horizon=2, damping=0.0, device=torch.device("cuda"),
                progress_solver="cem", cem_samples=16, cem_steps=2, cem_elite=4)
            rows.append({"seed": seed, "best_plan_equals_archived": row["best_plan_equals_archived"],
                         "result": result})
            print(json.dumps({"seed": seed, "physical": result["physical"]}), flush=True)
    finally:
        env.close()
    after = setup._parameter_hash(model)
    if before != after:
        raise RuntimeError("frozen model changed")
    report = {
        "scope": "known-root exploratory physical replay, not PC admission",
        "audit_sha256": setup._sha256_file(args.audit),
        "full_cem_sha256": setup._sha256_file(args.full_cem),
        "root_source_sha256": setup._sha256_file(args.root_source),
        "runner_sha256": setup._sha256_file(Path(__file__)),
        "parameter_hash_before": before, "parameter_hash_after": after,
        "cost_scope": "initial search charged from audit; initial search wall time not included",
        "rows": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
