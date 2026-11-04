"""Replay all fixed tangent actions against frozen-model RGB predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import run_reacher_pc_guarded_prediction_replay as replay
from experiments.spectral_reachability import run_reacher_pc_tangent_full_physical as physical
from experiments.spectral_reachability.reacher_d2p_context import advance_executed_context, restore_execution_snapshot
from experiments.spectral_reachability.reacher_d2p_goal_encoder_metric import GoalEncoderMetric

PHYSICAL_SHA = "2c3956da23c504f20b4b05d113e30c3f3055169c021e10e63b6732c963c90f74"


def branch_index(row):
    entries = {(b["prefix_index"], b["tail_index"]): b for b in row["branches"]}
    if len(row["branches"]) != 35 or set(entries) != {(i, j) for i in range(7) for j in range(5)}:
        raise ValueError("physical branch roster mismatch")
    if any(b["status"] != "complete" or b["completed_steps"] != 15 for b in entries.values()):
        raise ValueError("physical branch incomplete")
    return entries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-report", type=Path, required=True)
    p.add_argument("--physical", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    if (setup._sha256_file(args.model_report) != physical.MODEL_SHA
            or setup._sha256_file(args.physical) != PHYSICAL_SHA):
        raise RuntimeError("source report SHA mismatch")
    model_report, physical_report, preflight = (json.loads(path.read_text()) for path in
                                                 (args.model_report, args.physical, args.preflight))
    if (setup._sha256_file(args.preflight) != model_report["preflight_sha256"]
            or physical_report["model_sha256"] != physical.MODEL_SHA):
        raise RuntimeError("preflight or physical source binding mismatch")
    stats, weights = Path(preflight["action_stats_path"]), Path(preflight["weights_path"])
    if (setup._sha256_file(stats) != preflight["action_stats_sha256"]
            or setup._sha256_file(weights) != preflight["weights_sha256"]):
        raise RuntimeError("checkpoint or normalizer mismatch")
    from experiments.spectral_reachability import reacher_pc_p1_common_tail_preflight_20261102_codexexclusive as pf
    if setup._sha256_file(pf.Q_PATH) != preflight["q_report_sha256"]:
        raise RuntimeError("goal Q source mismatch")
    q = json.loads(pf.Q_PATH.read_text())
    setup.install_missing_field_compat()
    setup.validate_training_semantic_receipt(stats)
    mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
    _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
    model = setup.load_compatible_pretrained(weights.parent, device="cuda").eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    before = setup._parameter_hash(model)
    if before != preflight["model_parameter_hash_expected"]:
        raise RuntimeError("model parameter hash mismatch")
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "report.json"
    report = {"scope": "same-action RGB prediction replay on two exposed roots; no P1 admission",
              "physical_sha256": PHYSICAL_SHA, "model_report_sha256": physical.MODEL_SHA,
              "preflight_sha256": model_report["preflight_sha256"],
              "runner_sha256": setup._sha256_file(Path(__file__)), "parameter_hash_before": before,
              "cost_scope": "per completed branch: 3 open and 3 real-context refreshed token predictions; "
                            "multiple RGB encodes, inherited Q, and wall costs not included", "rows": []}
    print("Replay all fixed branches; compare open and refreshed predictions to real RGB encodings.", flush=True)
    try:
        seed_lists = [[r["seed"] for r in data["rows"]] for data in (model_report, physical_report, q)]
        if any(seeds != seed_lists[0] for seeds in seed_lists[1:]):
            raise RuntimeError("root roster mismatch")
        for mr, pr, qr, expected in zip(model_report["rows"], physical_report["rows"],
                                         q["rows"], preflight["roots"], strict=True):
            seed = mr["seed"]
            branches = branch_index(pr)
            prefixes, tails = physical.action_bank(mr, mean, std, env.action_space.low, env.action_space.high)
            packets, idx = setup.sample_history_packets(histories, seed=seed)
            root = setup.collect_executed_context(
                env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
            if (int(idx) != expected["history_index"] or root["root_rgb_sha256"] != expected["root_rgb_sha256"]
                    or root["goal_rgb_sha256"] != expected["goal_rgb_sha256"]):
                raise RuntimeError("root or goal binding mismatch")
            metric = GoalEncoderMetric(torch.as_tensor(qr["goal_metric"]["gram"], dtype=torch.float64, device="cuda"))
            pixels, goal, history = replay.prepare(processor, root, torch.device("cuda"))
            with torch.no_grad():
                goal_z = setup.p0._goal_latent(model, goal)
                target = setup.p0._planning_coordinate(model, goal_z, role="goal")
                root_embeddings = setup.p0._latent_context(model, pixels, history)
            row = {"seed": seed, "branches": []}
            report["rows"].append(row)
            for i in range(7):
                for j in range(5):
                    item = {"prefix_index": i, "tail_index": j, "status": "started", "checkpoints": []}
                    row["branches"].append(item)
                    try:
                        raw = np.concatenate((prefixes[i], tails[j]))
                        norm = np.concatenate((np.asarray(mr["prefix_normalized"][i], dtype=np.float32),
                                               np.asarray(mr["tail_normalized"][j], dtype=np.float32)))
                        tokens = torch.as_tensor(norm[None], device="cuda")
                        with torch.no_grad():
                            open_predictions = replay.direct_rollout(model, root_embeddings, history, tokens)[0]
                            predicted_endpoint = setup.p0._planning_coordinate(
                                model, open_predictions[-1:].contiguous(), role="predicted")[0]
                            if not np.allclose(predicted_endpoint.cpu().numpy(),
                                               np.asarray(mr["predicted_endpoints"][i][j]), atol=1e-5, rtol=1e-5):
                                raise RuntimeError("sealed predicted endpoint mismatch")
                        restored, restored_goal = restore_execution_snapshot(
                            env, seed=seed, snapshot=root["execution_snapshot"])
                        item["identity"] = setup.p0._verify_execution_identity(
                            restored, restored_goal, root["root_frame"], root["goal"])
                        context = root
                        verified = restored
                        for step in range(3):
                            pix, _, hist = replay.prepare(processor, context, torch.device("cuda"))
                            with torch.no_grad():
                                embeddings = setup.p0._latent_context(model, pix, hist)
                                refreshed = replay.direct_rollout(model, embeddings, hist, tokens[:, step:step + 1])[0, 0]
                            context = advance_executed_context(
                                env, context=context, raw_action_block=raw[step * 5:(step + 1) * 5],
                                action_block=5, verified_root_frame=verified)
                            verified = context["root_frame"]
                            pix_actual, _, hist_actual = replay.prepare(processor, context, torch.device("cuda"))
                            with torch.no_grad():
                                observed = setup.p0._latent_context(model, pix_actual, hist_actual)[0, -1]
                                oq = setup.p0._planning_coordinate(model, observed[None], role="predicted")[0]
                                pq = setup.p0._planning_coordinate(model, open_predictions[step:step + 1], role="predicted")[0]
                                rq = setup.p0._planning_coordinate(model, refreshed[None], role="predicted")[0]
                                def goal_cost(value):
                                    return float(metric.apply((value - target).double()).square().mean())
                            raw_env = env.unwrapped
                            qdist = setup.p0._qpos_distance(np.asarray(raw_env.env.physics.data.qpos),
                                                             np.asarray(raw_env.env.task.target_qpos))
                            expected_qdist = branches[(i, j)]["trajectory"][5 * (step + 1)]["qpos_distance"]
                            if not np.isclose(qdist, expected_qdist, rtol=0, atol=1e-8):
                                raise RuntimeError("physical qpos replay mismatch")
                            item["checkpoints"].append({
                                "step": 5 * (step + 1), "rgb_sha256": context["root_rgb_sha256"],
                                "qpos_distance": qdist, "observed_latent": observed.cpu().tolist(),
                                "open_latent": open_predictions[step].cpu().tolist(),
                                "refreshed_latent": refreshed.cpu().tolist(),
                                "open_mse": float((open_predictions[step] - observed).square().mean()),
                                "refreshed_mse": float((refreshed - observed).square().mean()),
                                "observed_goal_q": goal_cost(oq), "open_goal_q": goal_cost(pq),
                                "refreshed_goal_q": goal_cost(rq)})
                        item["status"] = "complete"
                    except Exception as exc:
                        item.update(status="error_or_censored", error_type=type(exc).__name__, error=str(exc)[:1000])
                    path.write_text(json.dumps(report, indent=2, sort_keys=True))
            print(json.dumps({"seed": seed, "complete": sum(x["status"] == "complete" for x in row["branches"]),
                              "branches": len(row["branches"])}), flush=True)
    finally:
        env.close()
        report["parameter_hash_after"] = setup._parameter_hash(model)
        report["frozen_parameters_unchanged"] = report["parameter_hash_after"] == before
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
    if not report["frozen_parameters_unchanged"]:
        raise RuntimeError("model parameters changed")


if __name__ == "__main__":
    main()
