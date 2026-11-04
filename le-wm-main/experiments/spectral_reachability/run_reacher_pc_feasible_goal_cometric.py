"""Score sealed development trajectories with goal-only encoder co-metric."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import reacher_d2p_goal_metric_runtime as metric_runtime


def score(metric, latent, goal):
    value = torch.as_tensor(latent, dtype=torch.float64, device=metric.device)
    target = torch.as_tensor(goal, dtype=torch.float64, device=metric.device)
    if value.shape != target.shape or value.ndim != 1:
        raise ValueError("latent/goal mismatch")
    return float(metric.apply(value - target).square().mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--root-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite output")
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    replay, generated, roots = (json.loads(p.read_text()) for p in
                                (args.replay, args.model_report, args.root_source))
    if (replay["input_model_sha256"] != setup._sha256_file(args.model_report)
            or replay["root_source_sha256"] != setup._sha256_file(args.root_source)
            or generated["source_report_sha256"] != setup._sha256_file(args.root_source)):
        raise RuntimeError("sealed report binding mismatch")
    policy, stats = Path(roots["protocol"]["policy"]), Path(roots["protocol"]["action_stats"])
    if (setup._sha256_file(policy / "weights.pt") != generated["weights_sha256"]
            or setup._sha256_file(stats) != generated["action_stats_sha256"]):
        raise RuntimeError("weights or normalization mismatch")
    if [r["seed"] for r in replay["rows"]] != [r["seed"] for r in generated["rows"]]:
        raise RuntimeError("seed mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    setup.install_missing_field_compat()
    setup.validate_training_semantic_receipt(stats)
    mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
    _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
    model = setup.load_compatible_pretrained(policy, device="cuda").eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    before = setup._parameter_hash(model)
    if before != generated["parameter_hash_before"] or before != replay["model_parameter_hash_before"]:
        raise RuntimeError("checkpoint tensor mismatch")
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    rows = []
    try:
        for observed, generated_row in zip(replay["rows"], generated["rows"], strict=True):
            seed = observed["seed"]
            packets, history_index = setup.sample_history_packets(histories, seed=seed)
            context = setup.collect_executed_context(
                env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
            if (int(history_index) != generated_row["history_index"]
                    or context["root_rgb_sha256"] != generated_row["root_rgb_sha256"]
                    or context["goal_rgb_sha256"] != generated_row["goal_rgb_sha256"]):
                raise RuntimeError("root, goal or history index mismatch")
            _, goal, _ = setup.p0._prepare(
                processor, context["frames"], context["root_frame"], context["goal"],
                np.asarray(context["history_action_blocks"], dtype=np.float32), torch.device("cuda"))
            target = setup.p0._goal_latent(model, goal)
            metric, metric_receipt = metric_runtime.measure_goal(
                model, goal, target.cpu().tolist(), seed=seed)
            arms = {}
            for arm, checkpoints in observed["arms"].items():
                arms[arm] = []
                for point in checkpoints:
                    costs = {}
                    for key in ("open_prediction", "refreshed_prediction", "observed"):
                        latent = point[key]["latent"]
                        difference = torch.as_tensor(latent, dtype=torch.float64) - target.cpu().double()
                        euclidean = float(difference.square().mean())
                        if not np.isclose(euclidean, point[key]["goal_mse"], rtol=1e-5, atol=1e-6):
                            raise RuntimeError("saved euclidean goal cost mismatch")
                        costs[key] = {"euclidean": euclidean, "goal_cometric": score(metric, latent, target)}
                    arms[arm].append({"physical_step": point["physical_step"],
                                      "qpos_distance": point["qpos_distance"], "costs": costs})
            rows.append({"seed": seed, "history_index": int(history_index),
                         "root_rgb_sha256": context["root_rgb_sha256"], "goal_rgb_sha256": context["goal_rgb_sha256"],
                         "goal_encoding": target.cpu().tolist(), "metric": metric_receipt, "arms": arms})
            print(json.dumps({"seed": seed, "arms": len(arms),
                              "vjp_rows": metric_receipt["input_vjp_rows"]}), flush=True)
    finally:
        env.close()
    after = setup._parameter_hash(model)
    if before != after:
        raise RuntimeError("frozen model changed")
    report = {"scope": "retrospective known-root co-metric diagnostic only",
              "replay_sha256": setup._sha256_file(args.replay),
              "model_report_sha256": setup._sha256_file(args.model_report),
              "root_source_sha256": setup._sha256_file(args.root_source),
              "script_sha256": setup._sha256_file(Path(__file__)),
              "metric_runtime_sha256": setup._sha256_file(Path(metric_runtime.__file__)),
              "model_before": before, "model_after": after,
              "encoder_cost": "per-root goal encoding plus metric receipt; no new model.predict calls",
              "rows": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
