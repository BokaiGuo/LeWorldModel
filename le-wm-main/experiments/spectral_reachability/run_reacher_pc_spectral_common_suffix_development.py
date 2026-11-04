"""Physical common-suffix intervention on already sealed development actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability.reacher_d2p_context import restore_execution_snapshot

ARMS = ("selected", "zero", "same_bank_greedy")
SUFFIXES = ("passive_zero", "selected_tail", "zero_tail", "same_bank_greedy_tail")


def fixed_actions(row, low, high):
    """Bind all three saved prefixes and all three saved tails without ranking."""
    low, high = np.asarray(low), np.asarray(high)
    prefixes, tails = {}, {}
    for arm in ARMS:
        saved = row["methods"][arm]
        prefix = np.asarray(saved["first_action_witness_raw"], dtype=np.float32)
        full = np.asarray(saved["action_witness_raw"], dtype=np.float32)
        if prefix.shape != (5, 2) or full.shape != (15, 2) or not np.array_equal(full[:5], prefix):
            raise ValueError(f"{arm}: invalid or unbound action witness")
        if not np.isfinite(full).all() or np.any(full < low - 1e-6) or np.any(full > high + 1e-6):
            raise ValueError(f"{arm}: action witness outside environment bounds")
        prefixes[arm] = prefix
        tails[f"{arm}_tail"] = full[5:].copy()
    tails["passive_zero"] = np.zeros((10, 2), dtype=np.float32)
    return prefixes, tails


def record(env):
    physics = env.unwrapped.env.physics
    task = env.unwrapped.env.task
    qpos = np.asarray(physics.data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(physics.data.qvel, dtype=np.float64).copy()
    target = np.asarray(task.target_qpos, dtype=np.float64).copy()
    return {"qpos": qpos.tolist(), "qvel": qvel.tolist(), "target_qpos": target.tolist(),
            "qpos_distance": setup.p0._qpos_distance(qpos, target)}


def execute(env, seed, context, actions):
    image, goal = restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"])
    identity = setup.p0._verify_execution_identity(image, goal, context["root_frame"], context["goal"])
    trajectory = [record(env)]
    stopped = None
    for index, action in enumerate(actions):
        _, reward, terminated, truncated, _ = env.step(action)
        state = record(env)
        state.update({"reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated)})
        trajectory.append(state)
        if terminated or truncated:
            stopped = {"step": index + 1, "terminated": bool(terminated), "truncated": bool(truncated)}
            break
    return {"identity": identity, "trajectory": trajectory, "stop": stopped,
            "completed_steps": len(trajectory) - 1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--physical", type=Path, required=True)
    p.add_argument("--model-report", type=Path, required=True)
    p.add_argument("--root-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite experiment output")
    physical, model_report, roots = (json.loads(path.read_text()) for path in
                                     (args.physical, args.model_report, args.root_source))
    if setup._sha256_file(args.model_report) != physical["input_report_sha256"]:
        raise RuntimeError("physical action source does not bind model report")
    if setup._sha256_file(args.root_source) != model_report["source_report_sha256"]:
        raise RuntimeError("model report does not bind root source")
    if [r["seed"] for r in physical["rows"]] != [r["seed"] for r in model_report["rows"]]:
        raise RuntimeError("physical and model seed lists differ")
    stats = Path(roots["protocol"]["action_stats"])
    if setup._sha256_file(stats) != model_report["action_stats_sha256"]:
        raise RuntimeError("action statistics mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    setup.install_missing_field_compat()
    setup.validate_training_semantic_receipt(stats)
    _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65),
    )
    rows = []
    try:
        for source, model in zip(physical["rows"], model_report["rows"], strict=True):
            seed = source["seed"]
            packets, _ = setup.sample_history_packets(histories, seed=seed)
            context = setup.collect_executed_context(
                env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history",
            )
            if context["root_rgb_sha256"] != model["root_rgb_sha256"] or context["goal_rgb_sha256"] != model["goal_rgb_sha256"]:
                raise RuntimeError("root or goal mismatch")
            prefixes, tails = fixed_actions(source, env.action_space.low, env.action_space.high)
            outcomes = {}
            for arm in ARMS:
                outcomes[arm] = {}
                for tail in SUFFIXES:
                    actions = np.concatenate([prefixes[arm], tails[tail]], axis=0)
                    result = execute(env, seed, context, actions)
                    # Verify the prefix replay against its own original physical receipt.
                    expected = source["methods"][arm]["physical"]["after_first_chunk_qpos_distance"]
                    if result["completed_steps"] < 5 or not np.isclose(
                            result["trajectory"][5]["qpos_distance"], expected, atol=1e-8, rtol=0):
                        raise RuntimeError("original prefix physical replay mismatch or early stop")
                    outcomes[arm][tail] = result
            rows.append({"seed": seed, "arms": outcomes})
            print(json.dumps({"seed": seed, "branches": 12,
                              "completed_full": sum(v["completed_steps"] == 15 for arm in outcomes.values() for v in arm.values())}),
                  flush=True)
    finally:
        env.close()
    report = {
        "scope": "exploratory same-action physical intervention on two known development roots",
        "tails_rule": "all three previously optimized tails plus passive zero; chosen irrespective of outcome",
        "caveat": "tail family is outcome-exposed, no independent generalization or gate admission",
        "input_physical_sha256": setup._sha256_file(args.physical),
        "input_model_sha256": setup._sha256_file(args.model_report),
        "input_roots_sha256": setup._sha256_file(args.root_source),
        "script_sha256": setup._sha256_file(Path(__file__)),
        "rows": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
