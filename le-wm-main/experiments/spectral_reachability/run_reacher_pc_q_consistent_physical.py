"""Real-RGB per-prefix Q-CEM on two exposed Reacher development roots."""
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
from experiments.spectral_reachability.run_pc_ap_reacher_rgb_physical_dev import ObserveSteps


class RestorableObserveSteps(ObserveSteps):
    """Forward the evaluator's goal restoration to the underlying wrapper."""

    @ObserveSteps._goal.setter
    def _goal(self, value):
        self.env._goal = value


def restore_paired_root(env, seed, context):
    """Restore and verify each arm's common physical root before its prefix."""
    frame, goal = setup.p0.restore_execution_snapshot(
        env, seed=int(seed), snapshot=context["execution_snapshot"])
    identity = setup.p0._verify_execution_identity(
        frame, goal, context["root_frame"], context["goal"])
    return frame, identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--q-source", type=Path, required=True)
    parser.add_argument("--root-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    if setup._sha256_file(args.source) != args.source_sha256:
        raise RuntimeError("sealed selection SHA mismatch")
    source = json.loads(args.source.read_text())
    q_source = json.loads(args.q_source.read_text())
    roots = json.loads(args.root_source.read_text())
    if (setup._sha256_file(args.q_source) != source["source_sha256"]
            or setup._sha256_file(args.root_source) != source["root_source_sha256"]):
        raise RuntimeError("Q or root source SHA mismatch")
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
        raise RuntimeError("frozen model identity mismatch")
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    env = RestorableObserveSteps(setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65)))
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "exposed development roots; no independent or P1 admission",
              "source_sha256": args.source_sha256, "q_source_sha256": source["source_sha256"],
              "root_source_sha256": source["root_source_sha256"],
              "runner_sha256": setup._sha256_file(Path(__file__)),
              "weights_sha256": source["weights_sha256"], "stats_sha256": source["action_stats_sha256"],
              "parameter_hash_before": before, "rows": [],
              "cost_scope": "suffix counts exclude inherited prefix, Q VJPs, encoder and setup costs"}
    path = args.output / "report.json"
    print("Execute selected, zero and Q-greedy; refresh real RGB then optimize Q suffix.", flush=True)
    try:
        setup.p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
        low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
        lower, upper = torch.as_tensor(low, device="cuda"), torch.as_tensor(high, device="cuda")
        q_rows = {r["seed"]: r for r in q_source["rows"]}
        if len(q_rows) != len(q_source["rows"]):
            raise ValueError("duplicate Q roots")
        for source_row in source["rows"]:
            seed = int(source_row["seed"])
            q_row = q_rows[seed]
            if (source_row["bank_params"] != q_row["bank_params"]
                    or source_row["root_rgb_sha256"] != q_row["root_rgb_sha256"]
                    or source_row["goal_rgb_sha256"] != q_row["goal_rgb_sha256"]):
                raise RuntimeError("Q and optimized source bank/root mismatch")
            selection = source_row["selection"]
            selected, eligible = chooser.select(selection["proposal_rows"])
            if (selected != selection["selected_proposal_index"]
                    or eligible != selection["eligible_count"]):
                raise RuntimeError("sealed selector mismatch")
            greedy = int(np.argmin([r["immediate_goal_cost"] for r in selection["proposal_rows"]]))
            indices = {"selected": 0 if selected is None else selected, "zero": 0,
                       "same_bank_q_greedy": greedy}
            packets, history_index = setup.sample_history_packets(histories, seed=seed)
            context = setup.collect_executed_context(
                env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
            if (int(history_index) != source_row["history_index"]
                    or context["root_rgb_sha256"] != source_row["root_rgb_sha256"]
                    or context["goal_rgb_sha256"] != source_row["goal_rgb_sha256"]):
                raise RuntimeError("physical root, goal or history mismatch")
            bank = torch.as_tensor(source_row["bank_params"], dtype=torch.float32, device="cuda")
            metric = GoalEncoderMetric(torch.as_tensor(q_row["goal_metric"]["gram"],
                                                       dtype=torch.float64, device="cuda"))
            root = {"seed": seed, "indices": indices, "selected_eligible": selected is not None,
                    "arms": {}}
            report["rows"].append(root)
            for arm, index in indices.items():
                env.events.clear()
                row = {"candidate_index": index, "status": "started",
                       "selected_zero_fallback": arm == "selected" and selected is None}
                root["arms"][arm] = row
                try:
                    restored_frame, root_identity = restore_paired_root(env, seed, context)
                    row["root_identity"] = root_identity
                    row["status"] = "root_restored"
                    params = bank[index:index + 1]
                    first_raw, first_normalized = setup.p0._to_raw_actions(
                        params, mean, std, lower, upper, env.action_space.low, env.action_space.high)
                    row.update(prefix_raw=first_raw.tolist(),
                               prefix_normalized=first_normalized.tolist(),
                               status="prefix_execution_started")
                    first = setup.advance_executed_context(
                        env, context=context, raw_action_block=first_raw, action_block=5,
                        verified_root_frame=restored_frame)
                    row.update(status="prefix_complete",
                               prefix_successor_rgb_sha256=first["root_rgb_sha256"],
                               advanced_action_sha256=first["advanced_action_sha256"])
                    raw_env = env.unwrapped
                    after_prefix = setup.p0._qpos_distance(
                        np.asarray(raw_env.env.physics.data.qpos),
                        np.asarray(raw_env.env.task.target_qpos))
                    row["after_prefix_qpos_distance"] = float(after_prefix)
                    pixels, goal, history = setup.p0._prepare(
                        processor, first["frames"], first["root_frame"], first["goal"],
                        np.asarray(first["history_action_blocks"], dtype=np.float32), torch.device("cuda"))
                    target = setup.p0._planning_coordinate(model, setup.p0._goal_latent(model, goal), role="goal")

                    def objective(candidates):
                        terminal = setup.p0._terminal_batch(
                            model, pixels, goal, history, candidates,
                            action_lower=lower, action_upper=upper)
                        coordinate = setup.p0._planning_coordinate(model, terminal, role="predicted")
                        return metric.apply((coordinate - target).double().T).square().mean(dim=0)

                    with torch.no_grad():
                        suffix, receipt = cem.optimize(
                            objective, horizon=2, action_dim=history.shape[-1], device=history.device,
                            seed=seed + 5000, samples=16, steps=2, elite=4)
                    suffix_raw, suffix_normalized = setup.p0._to_raw_actions(
                        suffix, mean, std, lower, upper, env.action_space.low, env.action_space.high)
                    execution = {}
                    row.update(status="suffix_execution_started",
                               suffix_raw=suffix_raw.tolist(), suffix_normalized=suffix_normalized.tolist(),
                               suffix_receipt=receipt, suffix_execution=execution)
                    physical, identity = setup.p0._execute(
                        env, seed, suffix_raw, expected_frame=first["root_frame"],
                        expected_goal=first["goal"], execution_snapshot=first["execution_snapshot"],
                        execution_receipt=execution)
                    if abs(float(physical["initial_qpos_distance"]) - float(after_prefix)) > 1e-8:
                        raise RuntimeError("observed successor state differs from suffix snapshot")
                    row.update(status="complete", physical=physical, successor_identity=identity)
                except Exception as exc:
                    row.update(failed_stage=row["status"], status="error_or_censored",
                               error_type=type(exc).__name__, error=str(exc)[:1000])
                finally:
                    row["events"] = list(env.events)
                    path.write_text(json.dumps(report, indent=2, sort_keys=True))
                print(json.dumps({"seed": seed, "arm": arm, "status": row["status"]}), flush=True)
    finally:
        env.close()
    after = setup._parameter_hash(model)
    report["parameter_hash_after"] = after
    report["frozen_parameters_unchanged"] = before == after
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    if before != after:
        raise RuntimeError("frozen parameters changed")


if __name__ == "__main__":
    main()
