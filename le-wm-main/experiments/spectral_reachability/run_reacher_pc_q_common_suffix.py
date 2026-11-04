"""Fixed common-suffix intervention on the sealed Q-feedback development run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import run_reacher_pc_spectral_common_suffix_development as common

ARMS = ("selected", "zero", "same_bank_q_greedy")
PHYSICAL_SHA = "0671a015c0b850ab41fc170817199c2658b5a57e5c5418de4042ffc32dd2474b"


def fixed_actions(row, low, high):
    if set(row["arms"]) != set(ARMS):
        raise ValueError("incomplete arm set")
    prefixes, tails = {}, {}
    for arm in ARMS:
        saved = row["arms"][arm]
        if saved["status"] != "complete" or saved["candidate_index"] != row["indices"][arm]:
            raise ValueError("incomplete or mismatched arm")
        prefix = np.asarray(saved["prefix_raw"], dtype=np.float32)
        tail = np.asarray(saved["suffix_raw"], dtype=np.float32)
        if prefix.shape != (5, 2) or tail.shape != (10, 2):
            raise ValueError("invalid action shape")
        actions = np.concatenate((prefix, tail))
        if (not np.isfinite(actions).all() or np.any(actions < np.asarray(low) - 1e-6)
                or np.any(actions > np.asarray(high) + 1e-6)):
            raise ValueError("invalid action bounds or values")
        digest = hashlib.sha256(np.ascontiguousarray(prefix).tobytes()).hexdigest()
        if digest != saved["advanced_action_sha256"]:
            raise ValueError("prefix action digest mismatch")
        prefixes[arm], tails[arm + "_tail"] = prefix, tail
    tails["passive_zero"] = np.zeros((10, 2), dtype=np.float32)
    if np.any(tails["passive_zero"] < low) or np.any(tails["passive_zero"] > high):
        raise ValueError("zero tail outside action bounds")
    return prefixes, tails


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--physical", type=Path, required=True)
    p.add_argument("--model-report", type=Path, required=True)
    p.add_argument("--root-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if setup._sha256_file(args.physical) != PHYSICAL_SHA:
        raise RuntimeError("sealed physical source mismatch")
    physical, model, roots = (json.loads(path.read_text()) for path in
                               (args.physical, args.model_report, args.root_source))
    if (setup._sha256_file(args.model_report) != physical["source_sha256"]
            or setup._sha256_file(args.root_source) != physical["root_source_sha256"]
            or [r["seed"] for r in physical["rows"]] != [r["seed"] for r in model["rows"]]):
        raise RuntimeError("source binding mismatch")
    stats = Path(roots["protocol"]["action_stats"])
    if setup._sha256_file(stats) != physical["stats_sha256"]:
        raise RuntimeError("action normalizer mismatch")
    setup.install_missing_field_compat()
    setup.validate_training_semantic_receipt(stats)
    _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "outcome-exposed development common-action intervention, not P1 admission",
              "physical_sha256": PHYSICAL_SHA, "model_sha256": physical["source_sha256"],
              "root_source_sha256": physical["root_source_sha256"],
              "runner_sha256": setup._sha256_file(Path(__file__)), "rows": []}
    output = args.output / "report.json"
    print("Measure each fixed prefix under every fixed ten-step suffix; no planner sees physical state.",
          flush=True)
    try:
        for source, model_row in zip(physical["rows"], model["rows"], strict=True):
            seed = source["seed"]
            packets, index = setup.sample_history_packets(histories, seed=seed)
            context = setup.collect_executed_context(
                env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
            if (int(index) != model_row["history_index"]
                    or context["root_rgb_sha256"] != model_row["root_rgb_sha256"]
                    or context["goal_rgb_sha256"] != model_row["goal_rgb_sha256"]):
                raise RuntimeError("root, goal or history mismatch")
            prefixes, tails = fixed_actions(source, env.action_space.low, env.action_space.high)
            row = {"seed": seed, "arms": {}, "tail_sha256": {
                name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                for name, value in tails.items()}}
            report["rows"].append(row)
            for arm in ARMS:
                row["arms"][arm] = {}
                for name, tail in tails.items():
                    try:
                        result = common.execute(env, seed, context, np.concatenate((prefixes[arm], tail)))
                        result["status"] = "complete" if result["completed_steps"] == 15 else "censored"
                        if result["completed_steps"] >= 5:
                            actual = result["trajectory"][5]["qpos_distance"]
                            expected = source["arms"][arm]["after_prefix_qpos_distance"]
                            result["prefix_replay_matches"] = bool(np.isclose(actual, expected, atol=1e-8, rtol=0))
                            if not result["prefix_replay_matches"]:
                                result["status"] = "prefix_mismatch"
                        row["arms"][arm][name] = result
                    except Exception as exc:
                        row["arms"][arm][name] = {"status": "error", "error_type": type(exc).__name__,
                                                   "error": str(exc)[:1000]}
                    output.write_text(json.dumps(report, indent=2, sort_keys=True))
            print(json.dumps({"seed": seed, "completed": sum(
                v["status"] == "complete" for a in row["arms"].values() for v in a.values())}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
