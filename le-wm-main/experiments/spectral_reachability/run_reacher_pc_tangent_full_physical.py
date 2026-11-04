"""Evaluate every sealed tangent prefix against every sealed common tail."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import run_reacher_pc_spectral_common_suffix_development as common
from experiments.spectral_reachability.reacher_d2p_context import restore_execution_snapshot

MODEL_SHA = "d85a12443483e2268a0849ad72b6afccf3b68804a536df06bb0f780620f41589"


def action_bank(row, mean, std, low, high):
    if row["status"] != "complete" or row["compiler_receipt"]["selected_proposal_index"] is not None:
        raise ValueError("unexpected source status or selection")
    prefixes = np.asarray(row["prefix_raw"], dtype=np.float32)
    tails = np.asarray(row["tail_raw"], dtype=np.float32)
    prefix_norm = np.asarray(row["prefix_normalized"], dtype=np.float32)
    tail_norm = np.asarray(row["tail_normalized"], dtype=np.float32)
    if (prefixes.shape != (7, 5, 2) or tails.shape != (5, 10, 2)
            or prefix_norm.shape != (7, 1, 10) or tail_norm.shape != (5, 2, 10)):
        raise ValueError("unexpected action array shape")
    for raw, normalized in ((prefixes, prefix_norm), (tails, tail_norm)):
        if (not np.isfinite(raw).all() or not np.isfinite(normalized).all()
                or np.any(raw < np.asarray(low) - 1e-6) or np.any(raw > np.asarray(high) + 1e-6)):
            raise ValueError("invalid physical action")
        for i in range(len(raw)):
            reconstructed = setup.p0._normalized_blocks_to_raw_actions(
                normalized[i], mean, std, action_dim=2)
            if not np.array_equal(raw[i], reconstructed):
                raise ValueError("raw and normalized actions differ")
    return prefixes, tails


def execute_branch(env, seed, context, actions):
    result = {"status": "started", "trajectory": [], "stop": None}
    try:
        image, goal = restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"])
        result["identity"] = setup.p0._verify_execution_identity(
            image, goal, context["root_frame"], context["goal"])
        result["trajectory"].append(common.record(env))
        for index, action in enumerate(actions):
            _, reward, terminated, truncated, _ = env.step(action)
            point = common.record(env)
            point.update(reward=float(reward), terminated=bool(terminated), truncated=bool(truncated))
            result["trajectory"].append(point)
            if terminated or truncated:
                result["stop"] = {"step": index + 1, "terminated": bool(terminated), "truncated": bool(truncated)}
                break
        result["status"] = "complete" if len(result["trajectory"]) == 16 else "censored"
    except Exception as exc:
        result.update(status="error", error_type=type(exc).__name__, error=str(exc)[:1000])
    result["completed_steps"] = max(0, len(result["trajectory"]) - 1)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if setup._sha256_file(args.model) != MODEL_SHA:
        raise RuntimeError("model report SHA mismatch")
    model = json.loads(args.model.read_text())
    if setup._sha256_file(args.preflight) != model["preflight_sha256"]:
        raise RuntimeError("preflight SHA mismatch")
    preflight = json.loads(args.preflight.read_text())
    stats = Path(preflight["action_stats_path"])
    if setup._sha256_file(stats) != preflight["action_stats_sha256"]:
        raise RuntimeError("normalizer SHA mismatch")
    if [r["seed"] for r in model["rows"]] != [r["seed"] for r in preflight["roots"]]:
        raise RuntimeError("root roster mismatch")
    setup.install_missing_field_compat()
    setup.validate_training_semantic_receipt(stats)
    mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
    _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "report.json"
    report = {"scope": "all fixed tangent prefixes and tails; development, no selected policy or P1 admission",
              "model_sha256": MODEL_SHA, "preflight_sha256": model["preflight_sha256"],
              "runner_sha256": setup._sha256_file(Path(__file__)), "rows": []}
    print("Execute all seven prefixes and five common tails per known root, without selection.", flush=True)
    try:
        for source, expected in zip(model["rows"], preflight["roots"], strict=True):
            seed = source["seed"]
            packets, index = setup.sample_history_packets(histories, seed=seed)
            context = setup.collect_executed_context(
                env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
            if (int(index) != expected["history_index"] or context["root_rgb_sha256"] != expected["root_rgb_sha256"]
                    or context["goal_rgb_sha256"] != expected["goal_rgb_sha256"]):
                raise RuntimeError("root identity mismatch")
            prefixes, tails = action_bank(source, mean, std, env.action_space.low, env.action_space.high)
            row = {"seed": seed, "branches": [], "prefix_sha256": hashlib.sha256(prefixes.tobytes()).hexdigest(),
                   "tail_sha256": hashlib.sha256(tails.tobytes()).hexdigest()}
            report["rows"].append(row)
            for prefix_index, prefix in enumerate(prefixes):
                for tail_index, tail in enumerate(tails):
                    result = execute_branch(env, seed, context, np.concatenate((prefix, tail)))
                    result.update(prefix_index=prefix_index, tail_index=tail_index)
                    row["branches"].append(result)
                    path.write_text(json.dumps(report, indent=2, sort_keys=True))
            print(json.dumps({"seed": seed, "branches": len(row["branches"]),
                              "complete": sum(x["status"] == "complete" for x in row["branches"])}), flush=True)
    finally:
        env.close()
        path.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
