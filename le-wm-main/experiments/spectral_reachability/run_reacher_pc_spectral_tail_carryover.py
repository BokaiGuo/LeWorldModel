"""Recover each sealed spectral arm's imagined suffix and test RGB carryover."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import reacher_p1_predicted_successor_cem as selector
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner
from experiments.spectral_reachability.reacher_pc_tail_carryover import cem_with_carried_tail


def saved_suffix(row, candidate, *, horizon, action_dim, device):
    """Read a sealed parameter witness, or return None for legacy reports."""
    proposal = row["selection"]["proposal_rows"][candidate]
    if proposal["proposal_index"] != candidate:
        raise ValueError("proposal index mismatch")
    if "suffix_parameters" not in proposal:
        return None
    suffix = torch.as_tensor(proposal["suffix_parameters"], dtype=torch.float32, device=device)
    if suffix.shape != (horizon, action_dim) or not torch.isfinite(suffix).all():
        raise ValueError("invalid saved suffix witness")
    return suffix.unsqueeze(0)


def score_saved_suffix(model, goal_latent, embeddings, actions, suffix, *, lower, upper):
    """Recompute the imagined score; this is additional billable inference."""
    with torch.no_grad():
        terminal = selector.p1._terminal_from_state(
            model, embeddings, actions, suffix, action_lower=lower, action_upper=upper)
        terminal = setup.p0._planning_coordinate(model, terminal, role="predicted")
        goal = setup.p0._planning_coordinate(model, goal_latent, role="goal")
        cost = (terminal - goal).square().mean(dim=1)
    if cost.shape != (1,) or not torch.isfinite(cost).all():
        raise ValueError("invalid saved suffix score")
    return float(cost[0])


def check_physical_source_equivalence(source, physical_source):
    """Bind new receipts to the exact plans used by the archived physical run."""
    for key in ("source_report_sha256", "weights_sha256", "action_stats_sha256",
                "parameter_hash_before", "parameter_hash_after", "protocol"):
        if source[key] != physical_source[key]:
            raise ValueError(f"compiler source mismatch: {key}")
    if len(source["rows"]) != len(physical_source["rows"]):
        raise ValueError("compiler row count mismatch")
    for new, old in zip(source["rows"], physical_source["rows"], strict=True):
        for key in ("seed", "root_rgb_sha256", "goal_rgb_sha256", "bank_params", "selected_params"):
            if new[key] != old[key]:
                raise ValueError(f"compiler row mismatch: {key}")
        a, b = new["selection"], old["selection"]
        if a["selected_proposal_index"] != b["selected_proposal_index"] or len(a["proposal_rows"]) != len(b["proposal_rows"]):
            raise ValueError("compiler selection mismatch")
        for x, y in zip(a["proposal_rows"], b["proposal_rows"], strict=True):
            for key in ("proposal_index", "immediate_goal_cost", "cem_future_goal_cost"):
                if x[key] != y[key]:
                    raise ValueError(f"compiler proposal mismatch: {key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--physical", required=True, type=Path)
    parser.add_argument("--physical-source", type=Path,
                        help="original compiler report bound by physical report, if different")
    parser.add_argument("--root-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    source, physical, roots = (json.loads(path.read_text()) for path in
                               (args.source, args.physical, args.root_source))
    physical_source = args.physical_source or args.source
    if (source["source_report_sha256"] != setup._sha256_file(args.root_source) or
            physical["input_report_sha256"] != setup._sha256_file(physical_source)):
        raise RuntimeError("report provenance mismatch")
    if args.physical_source:
        check_physical_source_equivalence(source, json.loads(args.physical_source.read_text()))
    policy, stats = Path(roots["protocol"]["policy"]), Path(roots["protocol"]["action_stats"])
    for key, path in (("weights_sha256", policy / "weights.pt"), ("action_stats_sha256", stats)):
        if setup._sha256_file(path) != source[key]:
            raise RuntimeError(f"{key} mismatch")
    if (source["protocol"]["samples"], source["protocol"]["steps"], source["protocol"]["elite"]) != (16, 2, 4):
        raise RuntimeError("unexpected source CEM budget")
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
        raise RuntimeError("checkpoint mismatch")
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
            old = next(r for r in physical["rows"] if r["seed"] == seed)
            root_row = next(r for r in roots["rows"] if r["seed"] == seed)
            packets, index = setup.sample_history_packets(histories, seed=seed)
            context = setup.collect_executed_context(
                env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
            if (int(index) != root_row["history_index"] or
                    context["root_rgb_sha256"] != row["root_rgb_sha256"] or
                    context["goal_rgb_sha256"] != row["goal_rgb_sha256"]):
                raise RuntimeError("root identity mismatch")
            pixels, goal, history = setup.p0._prepare(
                processor, context["frames"], context["root_frame"], context["goal"],
                np.asarray(context["history_action_blocks"], dtype=np.float32), torch.device("cuda"))
            bank = torch.as_tensor(row["bank_params"], dtype=torch.float32, device="cuda")
            goal_latent = setup.p0._goal_latent(model, goal)
            methods = {}
            for arm in ("selected", "zero", "same_bank_greedy"):
                candidate = old["indices"][arm]
                prefix = bank[candidate:candidate + 1]
                embedding, actions = setup._state_after_prefix(
                    model, pixels, history, prefix, action_lower=lower, action_upper=upper)
                archived_cost = row["selection"]["proposal_rows"][candidate]["cem_future_goal_cost"]
                tail = saved_suffix(row, candidate, horizon=2, action_dim=history.shape[-1], device="cuda")
                if tail is None:
                    tail, recovered = selector.cem_from_predicted_successor(
                        model, goal_latent, embedding, actions, future_horizon=2,
                        action_lower=lower, action_upper=upper, samples=16, steps=2, elite=4,
                        seed=seed + 5000)
                    recovered["witness_source"] = "legacy_reconstructed_cem"
                else:
                    recovered = {"best_cost": score_saved_suffix(
                        model, goal_latent, embedding, actions, tail, lower=lower, upper=upper),
                        "witness_source": "saved_suffix_parameters",
                        "forward_rollout_evaluations": 1,
                        "model_predict_sample_evaluations": int(tail.shape[1]),
                        "scope": "additional saved-witness verification; charge separately"}
                if not np.isclose(recovered["best_cost"], archived_cost, rtol=1e-6, atol=1e-7):
                    raise RuntimeError(f"{seed} {arm} imagined suffix score mismatch")

                def first(pixels_arg, goal_arg, history_arg, root_seed):
                    return prefix.clone(), {"selection": "sealed model prefix", "candidate_index": candidate}

                def carried(model_arg, pixels_arg, goal_arg, history_arg, **kw):
                    if (kw["future_horizon"], kw["samples"], kw["steps"], kw["elite"], kw["seed"]) != (
                            2, 16, 2, 4, seed + 5000):
                        raise ValueError("unexpected suffix contract")
                    def cost(candidates):
                        return setup.p0._cost_fn(
                            model_arg, pixels_arg, goal_arg, history_arg, candidates,
                            action_lower=kw["action_lower"], action_upper=kw["action_upper"])
                    return cem_with_carried_tail(cost, tail, samples=16, steps=2, elite=4, seed=kw["seed"])

                original_cem = runner.p0._cem
                try:
                    runner.p0._cem = carried
                    result = runner._two_stage(
                        name=arm, first_program=first, env=env, seed=seed, context=context,
                        preprocessor=processor, model=model, mean=mean, std=std, lower=lower, upper=upper,
                        action_block=5, progress_horizon=2, damping=0.0, device=torch.device("cuda"),
                        progress_solver="cem", cem_samples=16, cem_steps=2, cem_elite=4)
                finally:
                    runner.p0._cem = original_cem
                old_arm = old["methods"][arm]
                if (result["first_action_witness_raw"] != old_arm["first_action_witness_raw"] or
                        result["observed_successor_context"] != old_arm["observed_successor_context"]):
                    raise RuntimeError(f"{seed} {arm} physical prefix or successor mismatch")
                methods[arm] = {"candidate_index": candidate, "recovered_tail_params": tail.cpu().tolist(),
                                "recovered_imagined_receipt": recovered, "result": result}
                print(json.dumps({"seed": seed, "arm": arm, "physical": result["physical"],
                                  "selected_old_tail": result["suffix_receipt"]["selected_old_tail"]}), flush=True)
            rows.append({"seed": seed, "methods": methods})
    finally:
        env.close()
    after = setup._parameter_hash(model)
    if before != after:
        raise RuntimeError("model parameters changed")
    report = {
        "scope": "known-root development physical experiment, not a PC/P1 gate",
        "source_sha256": setup._sha256_file(args.source),
        "physical_sha256": setup._sha256_file(args.physical),
        "physical_source_sha256": setup._sha256_file(physical_source),
        "root_source_sha256": setup._sha256_file(args.root_source),
        "runner_sha256": setup._sha256_file(Path(__file__)),
        "parameter_hash_before": before, "parameter_hash_after": after,
        "cost_scope": "source selector and compilation costs must be charged; runtime recovery overhead separate",
        "rows": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
