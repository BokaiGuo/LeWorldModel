"""Frozen-model common-tail tangent development run on two sealed roots."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import reacher_pc_p1_common_tail_preflight_20261102_codexexclusive as preflight
from experiments.spectral_reachability import reacher_pc_p1_common_tail_tangent_20261102_codexexclusive as proposal
from experiments.spectral_reachability import reacher_pc_p1_common_tail_response_census_20261102_codexexclusive as response
from experiments.spectral_reachability.reacher_d2p_goal_encoder_metric import GoalEncoderMetric


def verify_preflight(saved, actual):
    for key in ("schema", "roots", "q_report_sha256", "root_source_sha256", "bank_report_sha256",
                "weights_sha256", "action_stats_sha256", "model_parameter_hash_expected",
                "proposal_code_sha256", "preflight_code_sha256"):
        if saved[key] != actual[key]:
            raise ValueError("preflight mismatch: " + key)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    saved = json.loads(args.preflight.read_text())
    verify_preflight(saved, preflight.material())
    q = json.loads(preflight.Q_PATH.read_text())
    roots = json.loads(preflight.SOURCE.read_text())
    setup.install_missing_field_compat()
    stats = Path(saved["action_stats_path"])
    setup.validate_training_semantic_receipt(stats)
    mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
    _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
    model = setup.load_compatible_pretrained(Path(saved["weights_path"]).parent, device="cuda").eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    before = setup._parameter_hash(model)
    if before != saved["model_parameter_hash_expected"]:
        raise RuntimeError("loaded frozen parameter hash mismatch")
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "report.json"
    report = {
        "scope": "model-only known-root development; no physical outcome or P1 admission",
        "preflight_sha256": setup._sha256_file(args.preflight),
        "runner_sha256": setup._sha256_file(Path(__file__)),
        "proposal_sha256": setup._sha256_file(Path(proposal.__file__)),
        "response_census_sha256": setup._sha256_file(Path(response.__file__)),
        "source_q_sha256": saved["q_report_sha256"], "source_root_sha256": saved["root_source_sha256"],
        "parameter_hash_before": before, "rows": [],
        "cost_scope": "nominal model samples exclude encodes and inherited Q construction",
    }
    print("Evaluate fixed tangent candidates and common-tail responses with frozen CUDA model.", flush=True)
    try:
        setup.p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
        low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
        lower, upper = torch.as_tensor(low, device="cuda"), torch.as_tensor(high, device="cuda")
        for qrow in q["rows"]:
            seed = int(qrow["seed"])
            row = {"seed": seed, "status": "started"}
            report["rows"].append(row)
            try:
                packets, history_index = setup.sample_history_packets(histories, seed=seed)
                context = setup.collect_executed_context(
                    env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                    warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
                if (int(history_index) != qrow["history_index"]
                        or context["root_rgb_sha256"] != qrow["root_rgb_sha256"]
                        or context["goal_rgb_sha256"] != qrow["goal_rgb_sha256"]):
                    raise RuntimeError("root, goal or history identity mismatch")
                row.update(history_index=int(history_index), root_rgb_sha256=context["root_rgb_sha256"],
                           goal_rgb_sha256=context["goal_rgb_sha256"])
                pixels, goal, history = setup.p0._prepare(
                    processor, context["frames"], context["root_frame"], context["goal"],
                    np.asarray(context["history_action_blocks"], dtype=np.float32), torch.device("cuda"))
                metric = GoalEncoderMetric(torch.as_tensor(qrow["goal_metric"]["gram"],
                                                           dtype=torch.float64, device="cuda"))
                torch.cuda.synchronize()
                started = time.perf_counter()
                bank, receipt = proposal.compile_reacher(
                    model, pixels, goal, history, metric, lower=lower, upper=upper)
                torch.cuda.synchronize()
                row.update(compile_seconds=time.perf_counter() - started, compiler_receipt=receipt,
                           bank_params=bank.detach().cpu().tolist(), status="compiled")
                tails = proposal.common_tails(history.new_zeros(10))
                started = time.perf_counter()
                endpoints, raw_prefix, normalized_prefix = [], [], []
                with torch.no_grad():
                    for candidate in bank:
                        emb, actions = setup._state_after_prefix(
                            model, pixels, history, candidate.unsqueeze(0), action_lower=lower, action_upper=upper)
                        terminal = setup._terminal_from_state(
                            model, emb.expand(5, -1, -1), actions.expand(5, -1, -1), tails,
                            action_lower=lower, action_upper=upper)
                        endpoints.append(setup.p0._planning_coordinate(model, terminal, role="predicted"))
                        raw, normalized = setup.p0._to_raw_actions(
                            candidate.unsqueeze(0), mean, std, lower, upper,
                            env.action_space.low, env.action_space.high)
                        raw_prefix.append(raw.tolist())
                        normalized_prefix.append(normalized.tolist())
                    target = setup.p0._planning_coordinate(
                        model, setup.p0._goal_latent(model, goal), role="goal")
                    costs = [proposal.q_costs(metric, values, target).cpu().tolist() for values in endpoints]
                    for i, values in enumerate(costs):
                        if not np.allclose(values, receipt["proposal_rows"][i]["final_costs"], rtol=1e-5, atol=1e-7):
                            raise RuntimeError("saved endpoint scores differ from compiler")
                    censuses = [response.census(metric, endpoints[0], value) for value in endpoints]
                    tail_raw, tail_normalized = [], []
                    for tail in tails:
                        raw, norm = setup.p0._to_raw_actions(
                            tail.unsqueeze(0), mean, std, lower, upper,
                            env.action_space.low, env.action_space.high)
                        tail_raw.append(raw.tolist())
                        tail_normalized.append(norm.tolist())
                torch.cuda.synchronize()
                row.update(status="complete", diagnostic_seconds=time.perf_counter() - started,
                           predicted_endpoints=[x.cpu().tolist() for x in endpoints],
                           recalculated_final_costs=costs, response_census=censuses,
                           prefix_raw=raw_prefix, prefix_normalized=normalized_prefix,
                           tail_raw=tail_raw, tail_normalized=tail_normalized,
                           diagnostic_nominal_samples={"prefix": 7, "suffix_token": 70})
            except Exception as exc:
                row.update(failed_stage=row["status"], status="error", error_type=type(exc).__name__,
                           error=str(exc)[:1000])
            path.write_text(json.dumps(report, indent=2, sort_keys=True))
            print(json.dumps({"seed": seed, "status": row["status"],
                              "selected": row.get("compiler_receipt", {}).get("selected_proposal_index")}),
                  flush=True)
    finally:
        env.close()
        report["parameter_hash_after"] = setup._parameter_hash(model)
        report["frozen_parameters_unchanged"] = report["parameter_hash_after"] == before
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
    if not report["frozen_parameters_unchanged"]:
        raise RuntimeError("model parameters changed")


if __name__ == "__main__":
    main()
