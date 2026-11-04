"""Known-root Q-consistent suffix optimization over every sealed prefix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import reacher_pc_goal_q_selection as chooser
from experiments.spectral_reachability import reacher_pc_q_suffix_cem as cem
from experiments.spectral_reachability.reacher_d2p_goal_encoder_metric import GoalEncoderMetric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--root-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required; CPU neural inference forbidden")
    if setup._sha256_file(args.source) != args.source_sha256:
        raise RuntimeError("source SHA mismatch")
    source = json.loads(args.source.read_text())
    roots = json.loads(args.root_source.read_text())
    if source["source_report_sha256"] != setup._sha256_file(args.root_source):
        raise RuntimeError("root source mismatch")
    policy, stats = Path(roots["protocol"]["policy"]), Path(roots["protocol"]["action_stats"])
    if (setup._sha256_file(policy / "weights.pt") != source["weights_sha256"]
            or setup._sha256_file(stats) != source["action_stats_sha256"]):
        raise RuntimeError("checkpoint or normalizer mismatch")
    setup.install_missing_field_compat()
    setup.validate_training_semantic_receipt(stats)
    mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
    _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
    model = setup.load_compatible_pretrained(policy, device="cuda").eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    before = setup._parameter_hash(model)
    if before != source["parameter_hash_before"] or before != source["parameter_hash_after"]:
        raise RuntimeError("source model identity mismatch")
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    print("Q-consistent suffix CEM on every fixed known-root prefix; no physical selection.", flush=True)
    try:
        setup.p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
        low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
        lower, upper = torch.as_tensor(low, device="cuda"), torch.as_tensor(high, device="cuda")
        for source_row in source["rows"]:
            seed = int(source_row["seed"])
            packets, history_index = setup.sample_history_packets(histories, seed=seed)
            context = setup.collect_executed_context(
                env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
            if (int(history_index) != source_row["history_index"]
                    or context["root_rgb_sha256"] != source_row["root_rgb_sha256"]
                    or context["goal_rgb_sha256"] != source_row["goal_rgb_sha256"]):
                raise RuntimeError("root, goal or history binding mismatch")
            pixels, goal, history = setup.p0._prepare(
                processor, context["frames"], context["root_frame"], context["goal"],
                np.asarray(context["history_action_blocks"], dtype=np.float32), torch.device("cuda"))
            target = setup.p0._goal_latent(model, goal)
            goal_coord = setup.p0._planning_coordinate(model, target, role="goal")
            metric = GoalEncoderMetric(torch.as_tensor(source_row["goal_metric"]["gram"],
                                                       dtype=torch.float64, device="cuda"))
            bank = torch.as_tensor(source_row["bank_params"], dtype=history.dtype, device="cuda")
            old_rows = source_row["selection"]["proposal_rows"]
            if (bank.ndim != 3 or bank.shape[1:] != (1, history.shape[-1])
                    or not torch.isfinite(bank).all() or len(old_rows) != len(bank)
                    or [r["proposal_index"] for r in old_rows] != list(range(len(bank)))):
                raise RuntimeError("fixed candidate bank mismatch")
            scored = []
            with torch.no_grad():
                for index in range(len(bank)):
                    emb, actions = setup._state_after_prefix(
                        model, pixels, history, bank[index:index + 1],
                        action_lower=lower, action_upper=upper)
                    immediate = setup.p0._planning_coordinate(model, emb[:, -1], role="predicted")[0]
                    immediate_cost = float(metric.apply((immediate - goal_coord).double()).square().mean())
                    if not np.isclose(immediate_cost, old_rows[index]["immediate_goal_cost"],
                                      rtol=1e-4, atol=1e-6):
                        raise RuntimeError(f"saved Q prefix score mismatch at {seed}/{index}")

                    def objective(candidates):
                        count = candidates.shape[0]
                        terminal = setup._terminal_from_state(
                            model, emb.expand(count, -1, -1), actions.expand(count, -1, -1),
                            candidates, action_lower=lower, action_upper=upper)
                        coordinate = setup.p0._planning_coordinate(model, terminal, role="predicted")
                        residual = (coordinate - goal_coord).double()
                        return metric.apply(residual.T).square().mean(dim=0)

                    suffix, receipt = cem.optimize(
                        objective, horizon=2, action_dim=history.shape[-1],
                        device=history.device, seed=seed + 5000,
                        samples=16, steps=2, elite=4)
                    scored.append({"proposal_index": index, "immediate_goal_cost": immediate_cost,
                                   "cem_future_goal_cost": receipt["best_final_cost"],
                                   "suffix_parameters": suffix[0].cpu().tolist(), "cem_receipt": receipt})
            selected, eligible = chooser.select(scored)
            rows.append({"seed": seed, "history_index": int(history_index),
                         "root_rgb_sha256": context["root_rgb_sha256"],
                         "goal_rgb_sha256": context["goal_rgb_sha256"],
                         "bank_params": source_row["bank_params"],
                         "selection": {"proposal_rows": scored, "selected_proposal_index": selected,
                                       "eligible_count": eligible, "reference": "neutral prefix"},
                         "selected_params": None if selected is None else bank[selected:selected + 1].cpu().tolist(),
                         "prior_q_selected": source_row["selection"]["selected_proposal_index"],
                         "inherited_goal_metric": {"encoder_forwards": source_row["goal_metric"]["encoder_forwards"],
                                                   "input_vjp_rows": source_row["goal_metric"]["input_vjp_rows"]},
                         "new_predict_samples": len(bank) * (1 + 2 * 32),
                         "new_predict_api_calls": len(bank) * (1 + 2 * 2)})
            print(json.dumps({"seed": seed, "selected": selected, "eligible": eligible,
                              "candidates": len(bank)}), flush=True)
    finally:
        env.close()
    after = setup._parameter_hash(model)
    if before != after:
        raise RuntimeError("frozen model parameters changed")
    report = {"scope": "known-root fixed-bank Q suffix development; no physical execution",
              "source_sha256": args.source_sha256,
              "root_source_sha256": setup._sha256_file(args.root_source),
              "weights_sha256": source["weights_sha256"], "action_stats_sha256": source["action_stats_sha256"],
              "parameter_hash_before": before, "parameter_hash_after": after,
              "runner_sha256": setup._sha256_file(Path(__file__)),
              "cem_sha256": setup._sha256_file(Path(cem.__file__)),
              "chooser_sha256": setup._sha256_file(Path(chooser.__file__)),
              "cost_scope": "new counts exclude inherited candidate compilation and goal Q VJP costs",
              "rows": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
