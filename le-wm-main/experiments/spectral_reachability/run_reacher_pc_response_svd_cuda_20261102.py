"""Sealed, known-root model-only response-SVD development run."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import reacher_pc_p1_common_tail_preflight_20261102_codexexclusive as preflight
from experiments.spectral_reachability import reacher_pc_p1_common_tail_tangent_20261102_codexexclusive as common
from experiments.spectral_reachability import reacher_pc_p1_constrained_common_tail_20261102_codexexclusive as strict
from experiments.spectral_reachability import reacher_pc_response_svd_20261102 as search
from experiments.spectral_reachability.reacher_d2p_goal_encoder_metric import GoalEncoderMetric
from experiments.spectral_reachability.run_reacher_pc_common_tangent_cuda import verify_preflight

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT.parent / "docs/PC_REACHER_RESPONSE_SVD_DEV_20261102.md"


def make_plan(preflight_path):
    saved = json.loads(preflight_path.read_text())
    verify_preflight(saved, preflight.material())
    sources = [Path(__file__), Path(search.__file__), Path(common.__file__),
               Path(strict.__file__), PROTOCOL]
    return {"schema": "pc_response_svd_development_v1",
            "preflight_sha256": setup._sha256_file(preflight_path),
            "sources": {str(path): setup._sha256_file(path) for path in sources},
            "weights_sha256": saved["weights_sha256"],
            "action_stats_sha256": saved["action_stats_sha256"],
            "model_parameter_hash": saved["model_parameter_hash_expected"],
            "roots": saved["roots"], "scope": "known roots; model only; no P1 admission"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        with args.plan.open("x") as stream:
            json.dump(make_plan(args.preflight), stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(json.dumps({"plan": str(args.plan), "sha256": setup._sha256_file(args.plan)}))
        return
    if args.output is None or args.output.exists() or not args.plan_sha256:
        raise ValueError("new output and sealed plan SHA required")
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    if setup._sha256_file(args.plan) != args.plan_sha256 or json.loads(args.plan.read_text()) != make_plan(args.preflight):
        raise RuntimeError("plan or source drift")
    saved = json.loads(args.preflight.read_text())
    q = json.loads(preflight.Q_PATH.read_text())
    if [int(row["seed"]) for row in q["rows"]] != [20263160, 20263161]:
        raise ValueError("unexpected root roster")
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
    report_path = args.output / "report.json"
    report = {"scope": "known-root model-only development; no P1 admission",
              "plan_sha256": args.plan_sha256, "parameter_hash_before": before, "rows": [],
              "cost_scope": "logical JVPs only; total backward, encoding and cost not certified"}
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
                metric = GoalEncoderMetric(torch.as_tensor(qr["goal_metric"]["gram"],
                                                           dtype=torch.float64, device="cuda"))
                torch.cuda.synchronize()
                start = time.perf_counter()
                bank, receipt = search.compile_reacher(model, pixels, goal, history, metric,
                                                       lower=lower, upper=upper)
                torch.cuda.synchronize()
                if len(bank) != len(receipt["proposal_rows"]):
                    raise RuntimeError("candidate count mismatch")
                raw, normalized = [], []
                for candidate in bank:
                    decoded, norm = setup.p0._to_raw_actions(
                        candidate.unsqueeze(0), mean, std, lower, upper,
                        env.action_space.low, env.action_space.high)
                    raw.append(decoded.tolist())
                    normalized.append(norm.tolist())
                row.update(status="complete", search_seconds=time.perf_counter() - start,
                           bank_params=bank.detach().cpu().tolist(), raw_actions=raw,
                           normalized_actions=normalized, compiler_receipt=receipt)
            except Exception as exc:
                row.update(status="error", error_type=type(exc).__name__, error=str(exc)[:1000])
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
            print(json.dumps({"seed": seed, "status": row["status"],
                              "selected": row.get("compiler_receipt", {}).get("strict_selected_index")}), flush=True)
    finally:
        env.close()
        report["parameter_hash_after"] = setup._parameter_hash(model)
        report["frozen_parameters_unchanged"] = report["parameter_hash_after"] == before
        report["plan_postcheck"] = (setup._sha256_file(args.plan) == args.plan_sha256
                                    and json.loads(args.plan.read_text()) == make_plan(args.preflight))
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not report["frozen_parameters_unchanged"] or not report["plan_postcheck"]:
        raise RuntimeError("model or source drift")


if __name__ == "__main__":
    if "--prepare" in sys.argv:
        main()
    else:
        from experiments.spectral_reachability.e3_p2_cooperative_gpu_lock_codex_20261030_v1 import gpu_lock
        with gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            main()
