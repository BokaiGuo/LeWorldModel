"""Frozen-model replay of already saved Reacher FD physical trajectories."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
import zlib
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_pc_p1_fd_common_physical_v2_20261102_codexexclusive as audit
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import reacher_pc_p1_common_tail_preflight_20261102_codexexclusive as pf
from experiments.spectral_reachability import run_reacher_pc_guarded_prediction_replay as replay
from experiments.spectral_reachability.reacher_d2p_goal_encoder_metric import GoalEncoderMetric


def decode_rgb(saved):
    shape = tuple(saved["shape"])
    if len(shape) != 3 or shape[-1] != 3 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("invalid RGB shape")
    data = zlib.decompress(base64.b64decode(saved["uint8_zlib_base64"], validate=True))
    if len(data) != int(np.prod(shape)) or hashlib.sha256(data).hexdigest() != saved["sha256"]:
        raise ValueError("RGB bytes or digest mismatch")
    return np.frombuffer(data, dtype=np.uint8).reshape(shape).copy()


def advance_saved(context, raw_block, successor):
    frames = context["frames"]
    blocks = context["history_action_blocks"]
    raw = np.asarray(raw_block, dtype=np.float32)
    if len(frames) != 3 or len(blocks) != 3 or raw.shape != (5, 2) or not np.isfinite(raw).all():
        raise ValueError("invalid causal action history")
    if successor.shape != context["root_frame"].shape or successor.dtype != np.uint8:
        raise ValueError("invalid successor RGB")
    return {"frames": [np.asarray(x).copy() for x in frames[1:]] + [context["root_frame"].copy()],
            "root_frame": successor.copy(), "goal": context["goal"].copy(),
            "history_action_blocks": [np.asarray(x).copy() for x in blocks[1:]] + [raw.reshape(-1).copy()]}


def main():
    p = argparse.ArgumentParser()
    for name in ("report", "plan", "model-report", "saved-audit", "preflight", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    for path, expected in ((args.report, audit.REPORT_SHA), (args.plan, audit.PLAN_SHA),
                           (args.model_report, audit.MODEL_SHA), (args.saved_audit, audit.SAVED_SHA)):
        audit.require(audit.sha(path) == expected, "input digest mismatch")
    report, plan, model_report, saved = [json.loads(x.read_text()) for x in
                                         (args.report, args.plan, args.model_report, args.saved_audit)]
    audit.verify(report, plan, model_report, saved)
    if any(audit.sha(path) != digest for path, digest in plan["sources"].items()):
        raise RuntimeError("sealed source digest mismatch")
    if audit.sha(args.preflight) != plan["preflight_sha256"]:
        raise RuntimeError("preflight mismatch")
    preflight = json.loads(args.preflight.read_text())
    stats, weights = Path(preflight["action_stats_path"]), Path(preflight["weights_path"])
    if (audit.sha(stats) != preflight["action_stats_sha256"] or
            audit.sha(weights) != preflight["weights_sha256"] or
            audit.sha(pf.Q_PATH) != preflight["q_report_sha256"]):
        raise RuntimeError("weights, normalizer or Q mismatch")
    q = json.loads(pf.Q_PATH.read_text())
    if [x["seed"] for x in q["rows"]] != [20263160, 20263161]:
        raise RuntimeError("Q root roster mismatch")
    setup.install_missing_field_compat()
    setup.validate_training_semantic_receipt(stats)
    mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
    _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
    model = setup.load_compatible_pretrained(weights.parent, device="cuda").eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    before = setup._parameter_hash(model)
    if before != preflight["model_parameter_hash_expected"]:
        raise RuntimeError("frozen model digest mismatch")
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "report.json"
    output = {"scope": "saved physical RGB diagnostic, exposed roots, no P1 admission",
              "status": "running", "physical_sha256": audit.REPORT_SHA, "plan_sha256": audit.PLAN_SHA,
              "model_sha256": audit.MODEL_SHA, "script_sha256": audit.sha(__file__),
              "parameter_hash_before": before, "rows": [], "expected_branches": 20,
              "costs": {"open_prediction_batches_attempted": 0, "refresh_prediction_batches_attempted": 0,
                        "context_encode_calls_attempted": 0}, "errors": []}
    start = time.monotonic()

    def save():
        path.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")

    print("Replay sealed physical RGB with the same model actions; no physical outcomes select actions.", flush=True)
    save()
    try:
        for mr, pr, qr, frozen in zip(model_report["rows"], report["rows"], q["rows"], plan["roots"], strict=True):
            seed = mr["seed"]
            packets, index = setup.sample_history_packets(histories, seed=seed)
            packets_array = np.asarray(packets, dtype=np.float32)
            expected_packets = np.asarray(frozen["warmup_packets"], dtype=np.float32)
            if (packets_array.shape != (3, 5, 2) or
                    not np.array_equal(packets_array, expected_packets) or
                    hashlib.sha256(np.ascontiguousarray(packets_array).tobytes()).hexdigest() != frozen["warmup_sha256"]):
                raise RuntimeError(f"warmup action mismatch for {seed}")
            root = setup.collect_executed_context(env, seed=seed, history_len=3, action_block=5,
                    warmup_packets=packets, warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
            if (int(index) != frozen["history_index"] or root["root_rgb_sha256"] != frozen["root_rgb_sha256"]
                    or root["goal_rgb_sha256"] != frozen["goal_rgb_sha256"]):
                raise RuntimeError(f"root identity mismatch for {seed}")
            metric = GoalEncoderMetric(torch.as_tensor(qr["goal_metric"]["gram"], dtype=torch.float64, device="cuda"))
            pixels, goal, history = replay.prepare(processor, root, torch.device("cuda"))
            with torch.no_grad():
                goal_z = setup.p0._goal_latent(model, goal)
                target = setup.p0._planning_coordinate(model, goal_z, role="goal")
                output["costs"]["context_encode_calls_attempted"] += 1
                embeddings = setup.p0._latent_context(model, pixels, history)
            row = {"seed": seed, "goal_coordinate": target.cpu().tolist(),
                   "goal_metric_source_sha256": preflight["q_report_sha256"], "branches": []}
            output["rows"].append(row)
            for b in pr["branches"]:
                item = {"arm": b["arm"], "prefix_index": b["prefix_index"],
                        "tail_index": b["tail_index"], "status": "started", "checkpoints": []}
                row["branches"].append(item)
                try:
                    i, j = b["prefix_index"], b["tail_index"]
                    raw = np.asarray(b["raw_actions"], dtype=np.float32)
                    norm = np.concatenate((np.asarray(mr["prefix_normalized"][i], dtype=np.float32),
                                           np.asarray(mr["tail_normalized"][j], dtype=np.float32)))
                    decoded = setup.p0._normalized_blocks_to_raw_actions(norm, mean, std, action_dim=2)
                    if raw.shape != (15, 2) or norm.shape != (3, 10) or not np.allclose(raw, decoded, atol=1e-6, rtol=0):
                        raise ValueError("raw-normalized action mismatch")
                    tokens = torch.as_tensor(norm[None], device="cuda")
                    with torch.no_grad():
                        output["costs"]["open_prediction_batches_attempted"] += 1
                        open_z = replay.direct_rollout(model, embeddings, history, tokens)[0]
                        endpoint = setup.p0._planning_coordinate(model, open_z[-1:].contiguous(), role="predicted")[0]
                        if not np.allclose(endpoint.cpu().numpy(), mr["predicted_endpoints"][i][j], atol=1e-5, rtol=1e-5):
                            raise ValueError("sealed endpoint mismatch")
                    context = root
                    for step in range(3):
                        pix, _, hist = replay.prepare(processor, context, torch.device("cuda"))
                        with torch.no_grad():
                            output["costs"]["context_encode_calls_attempted"] += 1
                            latent = setup.p0._latent_context(model, pix, hist)
                            output["costs"]["refresh_prediction_batches_attempted"] += 1
                            refreshed = replay.direct_rollout(model, latent, hist, tokens[:, step:step + 1])[0, 0]
                        record = b["trajectory"][5 * (step + 1)]
                        context = advance_saved(context, raw[step * 5:(step + 1) * 5], decode_rgb(record["rgb"]))
                        pix_actual, _, hist_actual = replay.prepare(processor, context, torch.device("cuda"))
                        with torch.no_grad():
                            output["costs"]["context_encode_calls_attempted"] += 1
                            observed = setup.p0._latent_context(model, pix_actual, hist_actual)[0, -1]
                            def cost(value, role):
                                projected = setup.p0._planning_coordinate(model, value[None], role=role)[0]
                                return (float(metric.apply((projected - target).double()).square().mean()),
                                        projected.cpu().tolist())
                            observed_q, observed_coordinate = cost(observed, "observed")
                            open_q, open_coordinate = cost(open_z[step], "predicted")
                            refreshed_q, refreshed_coordinate = cost(refreshed, "predicted")
                            item["checkpoints"].append({
                                "step": 5 * (step + 1), "rgb_sha256": record["rgb"]["sha256"],
                                "qpos_distance": record["qpos_distance"],
                                "observed_latent": observed.cpu().tolist(), "open_latent": open_z[step].cpu().tolist(),
                                "refreshed_latent": refreshed.cpu().tolist(),
                                "open_mse": float((open_z[step] - observed).square().mean()),
                                "refreshed_mse": float((refreshed - observed).square().mean()),
                                "observed_goal_q": observed_q, "open_goal_q": open_q, "refreshed_goal_q": refreshed_q,
                                "observed_coordinate": observed_coordinate, "open_coordinate": open_coordinate,
                                "refreshed_coordinate": refreshed_coordinate})
                    item["status"] = "complete"
                except Exception as exc:
                    item.update(status="error", error_type=type(exc).__name__, error=str(exc)[:1000])
                save()
            print(json.dumps({"seed": seed, "complete": sum(b["status"] == "complete" for b in row["branches"])}), flush=True)
        branches = [b for r in output["rows"] for b in r["branches"]]
        output["status"] = "complete" if len(branches) == 20 and all(b["status"] == "complete" for b in branches) else "partial"
    except Exception as exc:
        output["status"] = "error"
        output["errors"].append({"type": type(exc).__name__, "message": str(exc)[:1000]})
        raise
    finally:
        env.close()
        output["parameter_hash_after"] = setup._parameter_hash(model)
        output["frozen_parameters_unchanged"] = output["parameter_hash_after"] == before
        torch.cuda.synchronize()
        output["wall_seconds"] = time.monotonic() - start
        save()
    if not output["frozen_parameters_unchanged"]:
        raise RuntimeError("frozen model changed")


if __name__ == "__main__":
    main()
