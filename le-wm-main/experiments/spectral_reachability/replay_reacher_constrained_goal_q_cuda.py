"""Re-encode authenticated goal RGB to independently recompute saved Q scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import reacher_pc_p1_common_tail_preflight_20261102_codexexclusive as preflight
from experiments.spectral_reachability import audit_reacher_pc_constrained_saved as saved_audit
from experiments.spectral_reachability.reacher_d2p_goal_encoder_metric import GoalEncoderMetric
from experiments.spectral_reachability.run_reacher_pc_common_tangent_cuda import verify_preflight


def score_endpoints(metric, endpoints, target):
    if endpoints.ndim != 3 or endpoints.shape[:2] != (21, 5) or endpoints.shape[2] != target.numel():
        raise ValueError("endpoint shape")
    residual = (endpoints.reshape(-1, target.numel()) - target).double().T
    return metric.apply(residual).square().mean(dim=0).reshape(21, 5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    if setup._sha256_file(args.report) != saved_audit.SOURCE_SHA:
        raise RuntimeError("saved report SHA mismatch")
    source, pf = json.loads(args.report.read_text()), json.loads(args.preflight.read_text())
    verify_preflight(pf, preflight.material())
    q = json.loads(preflight.Q_PATH.read_text())
    if [r["seed"] for r in source["rows"]] != [r["seed"] for r in q["rows"]]:
        raise RuntimeError("root roster mismatch")
    setup.install_missing_field_compat()
    stats = Path(pf["action_stats_path"])
    setup.validate_training_semantic_receipt(stats)
    mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
    _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
    model = setup.load_compatible_pretrained(Path(pf["weights_path"]).parent, device="cuda").eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    before = setup._parameter_hash(model)
    if before != pf["model_parameter_hash_expected"]:
        raise RuntimeError("checkpoint parameters mismatch")
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "report.json"
    report = {"scope": "goal-only recomputation of saved model scores, no new action selection",
              "source_sha256": saved_audit.SOURCE_SHA, "preflight_sha256": setup._sha256_file(args.preflight),
              "runner_sha256": setup._sha256_file(Path(__file__)), "parameter_hash_before": before, "rows": []}
    print("Reconstruct real goal RGB and recalculate all fixed endpoint Q scores.", flush=True)
    try:
        for row, qr in zip(source["rows"], q["rows"], strict=True):
            seed = row["seed"]
            packets, index = setup.sample_history_packets(histories, seed=seed)
            context = setup.collect_executed_context(
                env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
            if (int(index) != row["history_index"] or context["root_rgb_sha256"] != row["root_rgb_sha256"]
                    or context["goal_rgb_sha256"] != row["goal_rgb_sha256"]):
                raise RuntimeError("goal, root or history mismatch")
            _, goal, _ = setup.p0._prepare(
                processor, context["frames"], context["root_frame"], context["goal"],
                np.asarray(context["history_action_blocks"], dtype=np.float32), torch.device("cuda"))
            metric = GoalEncoderMetric(torch.as_tensor(qr["goal_metric"]["gram"], dtype=torch.float64, device="cuda"))
            with torch.no_grad():
                target = setup.p0._planning_coordinate(model, setup.p0._goal_latent(model, goal), role="goal")
                endpoints = torch.as_tensor(row["predicted_endpoints"], dtype=target.dtype, device="cuda")
                actual = score_endpoints(metric, endpoints, target).cpu().numpy()
            expected = np.asarray([r["final_costs"] for r in row["compiler_receipt"]["proposal_rows"]])
            if expected.shape != (21, 5) or not np.isfinite(actual).all():
                raise ValueError("invalid score array")
            diff = np.abs(actual - expected)
            report["rows"].append({"seed": seed, "goal_rgb_sha256": context["goal_rgb_sha256"],
                                   "target_coordinate": target.cpu().tolist(), "scores": actual.tolist(),
                                   "max_absolute_score_error": float(diff.max()),
                                   "allclose_1e-7_plus_1e-5_relative": bool(np.allclose(actual, expected, atol=1e-7, rtol=1e-5)),
                                   "goal_encoder_forwards": 1})
            path.write_text(json.dumps(report, indent=2, sort_keys=True))
    finally:
        env.close()
        report["parameter_hash_after"] = setup._parameter_hash(model)
        report["frozen_parameters_unchanged"] = report["parameter_hash_after"] == before
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
    if not report["frozen_parameters_unchanged"] or not all(
            r["allclose_1e-7_plus_1e-5_relative"] for r in report["rows"]):
        raise RuntimeError("model changed or saved score mismatch")


if __name__ == "__main__":
    main()
