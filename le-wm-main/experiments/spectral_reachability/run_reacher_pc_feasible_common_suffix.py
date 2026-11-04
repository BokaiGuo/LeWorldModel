"""All-prefix/all-tail intervention on sealed known-root Reacher actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import run_reacher_pc_spectral_common_suffix_development as common

ARMS = ("selected", "zero", "same_bank_greedy", "refined_0", "refined_1")


def fixed_actions(row, low, high):
    low, high = np.asarray(low), np.asarray(high)
    prefixes, tails = {}, {}
    if set(row["methods"]) != set(ARMS):
        raise ValueError("incomplete fixed arm set")
    for arm in ARMS:
        saved = row["methods"][arm]
        prefix = np.asarray(saved["first_action_witness_raw"], dtype=np.float32)
        full = np.asarray(saved["action_witness_raw"], dtype=np.float32)
        if prefix.shape != (5, 2) or full.shape != (15, 2) or not np.array_equal(full[:5], prefix):
            raise ValueError("invalid action witness")
        if not np.isfinite(full).all() or np.any(full < low - 1e-6) or np.any(full > high + 1e-6):
            raise ValueError("action witness outside environment bounds")
        prefixes[arm] = prefix
        tails[f"{arm}_tail"] = full[5:].copy()
    tails["passive_zero"] = np.zeros((10, 2), dtype=np.float32)
    return prefixes, tails


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--root-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite")
    physical, model_report, roots = (json.loads(path.read_text()) for path in
                                     (args.physical, args.model_report, args.root_source))
    if setup._sha256_file(args.model_report) != physical["input_report_sha256"]:
        raise RuntimeError("physical and model reports not bound")
    if setup._sha256_file(args.root_source) != model_report["source_report_sha256"]:
        raise RuntimeError("model and root reports not bound")
    if [r["seed"] for r in physical["rows"]] != [r["seed"] for r in model_report["rows"]]:
        raise RuntimeError("seed mismatch")
    stats = Path(roots["protocol"]["action_stats"])
    if setup._sha256_file(stats) != model_report["action_stats_sha256"]:
        raise RuntimeError("action statistics mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    setup.install_missing_field_compat()
    setup.validate_training_semantic_receipt(stats)
    _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    rows = []
    try:
        for source, model in zip(physical["rows"], model_report["rows"], strict=True):
            seed = source["seed"]
            packets, index = setup.sample_history_packets(histories, seed=seed)
            context = setup.collect_executed_context(
                env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
            if (int(index) != model["history_index"]
                    or context["root_rgb_sha256"] != model["root_rgb_sha256"]
                    or context["goal_rgb_sha256"] != model["goal_rgb_sha256"]):
                raise RuntimeError("root, history or goal mismatch")
            prefixes, tails = fixed_actions(source, env.action_space.low, env.action_space.high)
            outcomes = {}
            for arm in ARMS:
                outcomes[arm] = {}
                for tail, tail_actions in tails.items():
                    result = common.execute(env, seed, context, np.concatenate([prefixes[arm], tail_actions]))
                    expected = source["methods"][arm]["physical"]["after_first_chunk_qpos_distance"]
                    if result["completed_steps"] < 5 or not np.isclose(
                            result["trajectory"][5]["qpos_distance"], expected, atol=1e-8, rtol=0):
                        raise RuntimeError("prefix replay mismatch or early stop")
                    outcomes[arm][tail] = result
            rows.append({"seed": seed, "arms": outcomes})
            print(json.dumps({"seed": seed, "branches": 30,
                              "completed_full": sum(v["completed_steps"] == 15
                                                    for arm in outcomes.values() for v in arm.values())}), flush=True)
    finally:
        env.close()
    report = {"scope": "known-root common-action intervention; no independent admission",
              "tails_rule": "five sealed tails plus zero, all used for all prefixes",
              "input_physical_sha256": setup._sha256_file(args.physical),
              "input_model_sha256": setup._sha256_file(args.model_report),
              "input_roots_sha256": setup._sha256_file(args.root_source),
              "script_sha256": setup._sha256_file(Path(__file__)), "rows": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
