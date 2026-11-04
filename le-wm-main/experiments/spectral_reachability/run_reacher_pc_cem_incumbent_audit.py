"""Replay the sealed full-CEM development inputs with an incumbent ledger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability.reacher_pc_cem_incumbent_audit import cem_incumbent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-cem", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--root-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    full, source, roots = (json.loads(p.read_text()) for p in
                           (args.full_cem, args.model_source, args.root_source))
    if (full["source_sha256"] != setup._sha256_file(args.model_source) or
            full["root_source_sha256"] != setup._sha256_file(args.root_source) or
            source["source_report_sha256"] != setup._sha256_file(args.root_source)):
        raise RuntimeError("report provenance mismatch")
    policy = Path(roots["protocol"]["policy"])
    stats = Path(roots["protocol"]["action_stats"])
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
        raise RuntimeError("checkpoint parameter mismatch")
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    rows = []
    try:
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
            pixels, goal, history = setup.p0._prepare(
                processor, context["frames"], context["root_frame"], context["goal"],
                np.asarray(context["history_action_blocks"], dtype=np.float32), torch.device("cuda"))
            def cost(candidates):
                return setup.p0._cost_fn(model, pixels, goal, history, candidates,
                                         action_lower=lower, action_upper=upper)
            best, final, receipt = cem_incumbent(
                cost, horizon=3, action_dim=history.shape[-1], device="cuda",
                samples=16, steps=12, elite=4, seed=seed + 9000)
            archived = torch.as_tensor(old["result"]["first_receipt"]["full_pre_feedback_plan_params"],
                                       device="cuda", dtype=final.dtype)
            if not torch.equal(final, archived):
                raise RuntimeError(f"archived final plan failed exact replay for seed {seed}")
            rows.append({"seed": seed, "root_rgb_sha256": context["root_rgb_sha256"],
                         "goal_rgb_sha256": context["goal_rgb_sha256"],
                         "best_plan_params": best.cpu().tolist(),
                         "archived_plan_exact_replay": True,
                         "best_plan_equals_archived": bool(torch.equal(best, archived)),
                         **receipt})
            print(json.dumps({"seed": seed, **receipt}), flush=True)
    finally:
        env.close()
    after = setup._parameter_hash(model)
    if before != after:
        raise RuntimeError("model parameters changed")
    report = {
        "scope": "development only; same-input model diagnostic, no physical result",
        "full_cem_sha256": setup._sha256_file(args.full_cem),
        "model_source_sha256": setup._sha256_file(args.model_source),
        "root_source_sha256": setup._sha256_file(args.root_source),
        "runner_sha256": setup._sha256_file(Path(__file__)),
        "parameter_hash_before": before, "parameter_hash_after": after, "rows": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
