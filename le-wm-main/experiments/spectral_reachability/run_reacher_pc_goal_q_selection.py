"""Select from a sealed Reacher bank using goal-encoder Q on CUDA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import reacher_d2p_goal_metric_runtime as metric_runtime
from experiments.spectral_reachability import reacher_pc_goal_q_selection as chooser


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
    source, roots = (json.loads(path.read_text()) for path in (args.source, args.root_source))
    if source["source_report_sha256"] != setup._sha256_file(args.root_source):
        raise RuntimeError("root source mismatch")
    policy, stats = Path(roots["protocol"]["policy"]), Path(roots["protocol"]["action_stats"])
    if (setup._sha256_file(policy / "weights.pt") != source["weights_sha256"]
            or setup._sha256_file(stats) != source["action_stats_sha256"]):
        raise RuntimeError("model source mismatch")
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
        raise RuntimeError("frozen parameter mismatch")
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    rows = []
    try:
        setup.p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
        low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
        lower, upper = torch.as_tensor(low, device="cuda"), torch.as_tensor(high, device="cuda")
        for source_row in source["rows"]:
            seed = source_row["seed"]
            packets, history_index = setup.sample_history_packets(histories, seed=seed)
            context = setup.collect_executed_context(
                env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
            if (int(history_index) != source_row["history_index"]
                    or context["root_rgb_sha256"] != source_row["root_rgb_sha256"]
                    or context["goal_rgb_sha256"] != source_row["goal_rgb_sha256"]):
                raise RuntimeError("root, goal or history index mismatch")
            pixels, goal, history = setup.p0._prepare(
                processor, context["frames"], context["root_frame"], context["goal"],
                np.asarray(context["history_action_blocks"], dtype=np.float32), torch.device("cuda"))
            target = setup.p0._goal_latent(model, goal)
            metric, metric_receipt = metric_runtime.measure_goal(model, goal, target.cpu().tolist(), seed=seed)
            bank = torch.as_tensor(source_row["bank_params"], dtype=history.dtype, device="cuda")
            saved = source_row["selection"]["proposal_rows"]
            if len(saved) != len(bank) or [x["proposal_index"] for x in saved] != list(range(len(bank))):
                raise RuntimeError("fixed bank and suffix rows not aligned")
            scored = []
            with torch.no_grad():
                for i, row in enumerate(saved):
                    suffix = torch.as_tensor(row["suffix_parameters"], dtype=history.dtype, device="cuda").unsqueeze(0)
                    if suffix.shape != (1, 2, history.shape[-1]) or not torch.isfinite(suffix).all():
                        raise RuntimeError("invalid saved suffix")
                    embeddings, actions = setup._state_after_prefix(
                        model, pixels, history, bank[i:i+1], action_lower=lower, action_upper=upper)
                    immediate = setup.p0._planning_coordinate(model, embeddings[:, -1], role="predicted")[0]
                    terminal = setup._terminal_from_state(
                        model, embeddings, actions, suffix, action_lower=lower, action_upper=upper)
                    final = setup.p0._planning_coordinate(model, terminal, role="predicted")[0]
                    goal_coordinate = setup.p0._planning_coordinate(model, target, role="goal")
                    prefix_cost = float(metric.apply((immediate - goal_coordinate).double()).square().mean())
                    final_cost = float(metric.apply((final - goal_coordinate).double()).square().mean())
                    scored.append({"proposal_index": i, "immediate_goal_cost": prefix_cost,
                                   "cem_future_goal_cost": final_cost,
                                   "suffix_parameters": row["suffix_parameters"]})
            selected, eligible = chooser.select(scored)
            rows.append({"seed": seed, "history_index": int(history_index),
                         "root_rgb_sha256": context["root_rgb_sha256"],
                         "goal_rgb_sha256": context["goal_rgb_sha256"],
                         "bank_params": source_row["bank_params"],
                         "original_bank_size": source_row["original_bank_size"],
                         "refinement": source_row["refinement"],
                         "selection": {"proposal_rows": scored, "selected_proposal_index": selected,
                                       "eligible_count": eligible, "reference": "neutral prefix"},
                         "selected_params": None if selected is None else bank[selected:selected+1].cpu().tolist(),
                         "goal_metric": metric_receipt,
                         "scoring_predict_samples": 3 * len(bank),
                         "prior_source_selection": source_row["selection"]["selected_proposal_index"]})
            print(json.dumps({"seed": seed, "selected": selected, "eligible": eligible}), flush=True)
    finally:
        env.close()
    after = setup._parameter_hash(model)
    if before != after:
        raise RuntimeError("frozen parameters changed")
    report = {"scope": "outcome-informed known-root model-only Q selector development",
              "source_report_sha256": setup._sha256_file(args.root_source),
              "fixed_bank_sha256": setup._sha256_file(args.source),
              "weights_sha256": source["weights_sha256"], "action_stats_sha256": source["action_stats_sha256"],
              "parameter_hash_before": before, "parameter_hash_after": after,
              "script_sha256": setup._sha256_file(Path(__file__)),
              "chooser_sha256": setup._sha256_file(Path(chooser.__file__)),
              "metric_runtime_sha256": setup._sha256_file(Path(metric_runtime.__file__)),
              "protocol": source["protocol"], "rows": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
