"""Sealed fixed-budget constrained search; known roots, model-only CUDA."""
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
from experiments.spectral_reachability import reacher_pc_p1_common_tail_tangent_20261102_codexexclusive as prior
from experiments.spectral_reachability import reacher_pc_p1_fixed_lattice_common_tail_20261102_codexexclusive as search
from experiments.spectral_reachability import reacher_pc_p1_common_tail_response_census_20261102_codexexclusive as census
from experiments.spectral_reachability.reacher_d2p_goal_encoder_metric import GoalEncoderMetric
from experiments.spectral_reachability.run_reacher_pc_common_tangent_cuda import verify_preflight

ROOT = Path(__file__).resolve().parents[2]
PROTOCOLS = (ROOT.parent / "docs/PC_REACHER_P1_FIXED_LATTICE_COMMON_TAIL_DEV_20261102_CODEXEXCLUSIVE.md",
             ROOT.parent / "docs/PC_REACHER_FIXED_LATTICE_CUDA_LAUNCH_20261102.md")


def plan(preflight_path):
    saved = json.loads(preflight_path.read_text())
    verify_preflight(saved, preflight.material())
    files = [Path(__file__), Path(search.__file__), Path(prior.__file__),
             Path(census.__file__), *PROTOCOLS]
    return {"schema": "pc_fixed_lattice_cuda_development_v1", "preflight_sha256": setup._sha256_file(preflight_path),
            "sources": {str(path): setup._sha256_file(path) for path in files},
            "q_sha256": saved["q_report_sha256"], "root_source_sha256": saved["root_source_sha256"],
            "weights_sha256": saved["weights_sha256"], "stats_sha256": saved["action_stats_sha256"],
            "parameters_sha256": saved["model_parameter_hash_expected"], "roots": saved["roots"],
            "scope": "two exposed roots; model-only; P1 admission false"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if args.plan.exists():
            raise FileExistsError(args.plan)
        data = plan(args.preflight)
        with args.plan.open("x") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        print(json.dumps({"plan": str(args.plan), "sha256": setup._sha256_file(args.plan)}))
        return
    if args.output is None or args.output.exists() or not args.plan_sha256:
        raise ValueError("new output and sealed plan SHA required")
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    if setup._sha256_file(args.plan) != args.plan_sha256 or json.loads(args.plan.read_text()) != plan(args.preflight):
        raise RuntimeError("plan or source drift")
    saved = json.loads(args.preflight.read_text())
    q = json.loads(preflight.Q_PATH.read_text())
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
        raise RuntimeError("model parameter hash mismatch")
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "report.json"
    report = {"scope": "known-root fixed lattice model-only development; no P1 admission",
              "plan_sha256": args.plan_sha256, "parameter_hash_before": before,
              "cost_scope": "logical samples exclude encoding, inherited Q and internal backward work", "rows": []}
    print("Run fixed lattice and non-gating common-tail census on both exposed roots.", flush=True)
    try:
        setup.p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
        low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
        lower, upper = torch.as_tensor(low, device="cuda"), torch.as_tensor(high, device="cuda")
        for qr in q["rows"]:
            seed = int(qr["seed"])
            row = {"seed": seed, "status": "started"}
            report["rows"].append(row)
            try:
                packets, index = setup.sample_history_packets(histories, seed=seed)
                context = setup.collect_executed_context(
                    env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                    warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
                if (int(index) != qr["history_index"] or context["root_rgb_sha256"] != qr["root_rgb_sha256"]
                        or context["goal_rgb_sha256"] != qr["goal_rgb_sha256"]):
                    raise RuntimeError("root, goal or history mismatch")
                row.update(history_index=int(index), root_rgb_sha256=context["root_rgb_sha256"],
                           goal_rgb_sha256=context["goal_rgb_sha256"])
                pixels, goal, history = setup.p0._prepare(
                    processor, context["frames"], context["root_frame"], context["goal"],
                    np.asarray(context["history_action_blocks"], dtype=np.float32), torch.device("cuda"))
                metric = GoalEncoderMetric(torch.as_tensor(qr["goal_metric"]["gram"], dtype=torch.float64, device="cuda"))
                torch.cuda.synchronize()
                start = time.perf_counter()
                bank, receipt = search.compile_reacher(model, pixels, goal, history, metric, lower=lower, upper=upper)
                torch.cuda.synchronize()
                if len(bank) != 41 or len(receipt["proposal_rows"]) != 41:
                    raise RuntimeError("candidate count mismatch")
                row.update(status="compiled", search_seconds=time.perf_counter() - start,
                           bank_params=bank.detach().cpu().tolist(), compiler_receipt=receipt)
                tails = prior.common_tails(history.new_zeros(10))
                endpoints, raw_prefix, normalized_prefix = [], [], []
                start = time.perf_counter()
                with torch.no_grad():
                    for candidate in bank:
                        emb, actions = setup._state_after_prefix(
                            model, pixels, history, candidate.unsqueeze(0), action_lower=lower, action_upper=upper)
                        final = setup._terminal_from_state(model, emb.expand(5, -1, -1), actions.expand(5, -1, -1),
                                                           tails, action_lower=lower, action_upper=upper)
                        endpoints.append(setup.p0._planning_coordinate(model, final, role="predicted"))
                        raw, norm = setup.p0._to_raw_actions(candidate.unsqueeze(0), mean, std, lower, upper,
                                                             env.action_space.low, env.action_space.high)
                        raw_prefix.append(raw.tolist())
                        normalized_prefix.append(norm.tolist())
                    target = setup.p0._planning_coordinate(model, setup.p0._goal_latent(model, goal), role="goal")
                    costs = [prior.q_costs(metric, values, target).cpu().tolist() for values in endpoints]
                    for i, values in enumerate(costs):
                        if not np.allclose(values, receipt["proposal_rows"][i]["final_costs"], rtol=1e-5, atol=1e-7):
                            raise RuntimeError("replayed endpoint Q scores differ")
                    spectra = [census.census(metric, endpoints[0], value) for value in endpoints]
                    raw_tails, normalized_tails = [], []
                    for tail in tails:
                        raw, norm = setup.p0._to_raw_actions(tail.unsqueeze(0), mean, std, lower, upper,
                                                             env.action_space.low, env.action_space.high)
                        raw_tails.append(raw.tolist())
                        normalized_tails.append(norm.tolist())
                torch.cuda.synchronize()
                row.update(status="complete", census_seconds=time.perf_counter() - start,
                           predicted_endpoints=[value.cpu().tolist() for value in endpoints],
                           recalculated_final_costs=costs, response_census=spectra, prefix_raw=raw_prefix,
                           prefix_normalized=normalized_prefix, tail_raw=raw_tails, tail_normalized=normalized_tails,
                           census_nominal_samples={"prefix": 41, "suffix_token": 410})
            except Exception as exc:
                row.update(failed_stage=row["status"], status="error", error_type=type(exc).__name__,
                           error=str(exc)[:1000])
            path.write_text(json.dumps(report, indent=2, sort_keys=True))
            print(json.dumps({"seed": seed, "status": row["status"],
                              "selected": row.get("compiler_receipt", {}).get("selected_proposal_index")}), flush=True)
    finally:
        env.close()
        report["parameter_hash_after"] = setup._parameter_hash(model)
        report["frozen_parameters_unchanged"] = report["parameter_hash_after"] == before
        report["plan_postcheck"] = (setup._sha256_file(args.plan) == args.plan_sha256
                                    and json.loads(args.plan.read_text()) == plan(args.preflight))
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
    if not report["frozen_parameters_unchanged"] or not report["plan_postcheck"]:
        raise RuntimeError("model or source drift")


if __name__ == "__main__":
    main()
