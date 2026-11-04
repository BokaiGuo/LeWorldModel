"""Exhaustive known-root physical panel for the fixed response-SVD bank."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
from experiments.spectral_reachability import reacher_d2p_context as context_lib
from experiments.spectral_reachability import reacher_pc_p1_common_tail_preflight_20261102_codexexclusive as preflight
from experiments.spectral_reachability import run_reacher_pc_common_tangent_cuda as prior_runner
from experiments.spectral_reachability import run_reacher_pc_p1_fd_common_physical_20261102_codexexclusive as physical
from experiments.spectral_reachability import run_reacher_pc_response_svd_cuda_20261102 as source_runner

SOURCE_SHA = "b3f4e0cab1c63c0c1f3d7c6cc43309a2d81e6644ea8748885569f38b097479ca"
MODEL_PLAN_SHA = "8c0d1a3decc124e12a29068289e49fe7597fe3286fc3e8b3a69cbcac31ff24d1"
ROOTS = (20263160, 20263161)
PROTOCOL = Path(__file__).resolve().parents[3] / "docs/PC_REACHER_RESPONSE_SVD_PHYSICAL_PROTOCOL_20261102.md"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode(params, mean, std, lower, upper):
    raw, _ = setup.p0._to_raw_actions(
        torch.as_tensor(params, dtype=torch.float32).unsqueeze(0), mean, std,
        torch.as_tensor(lower), torch.as_tensor(upper),
        np.asarray([-1., -1.], dtype=np.float32), np.asarray([1., 1.], dtype=np.float32))
    return np.asarray(raw, dtype=np.float32)


def build(source_path, source_plan_path, preflight_path):
    require(setup._sha256_file(source_path) == SOURCE_SHA, "source report drift")
    require(setup._sha256_file(source_plan_path) == MODEL_PLAN_SHA, "model plan drift")
    require(json.loads(source_plan_path.read_text()) == source_runner.make_plan(preflight_path),
            "model plan sources drift")
    source = json.loads(source_path.read_text())
    saved = json.loads(preflight_path.read_text())
    prior_runner.verify_preflight(saved, preflight.material())
    require(source["plan_sha256"] == MODEL_PLAN_SHA and source["plan_postcheck"]
            and source["frozen_parameters_unchanged"], "model postchecks")
    require(tuple(row["seed"] for row in source["rows"]) == ROOTS, "source roster")
    stats = Path(saved["action_stats_path"])
    setup.validate_training_semantic_receipt(stats)
    mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
    _, histories, _, history_receipt = setup.load_runtime_support(stats, history_len=3, action_block=5)
    lower, upper = setup._normalized_action_bounds(
        mean, std, np.asarray([-1., -1.], dtype=np.float32),
        np.asarray([1., 1.], dtype=np.float32))
    roots = []
    for model_row, root in zip(source["rows"], saved["roots"], strict=True):
        require(model_row["seed"] == root["seed"] and model_row["status"] == "complete"
                and model_row["root_rgb_sha256"] == root["root_rgb_sha256"]
                and model_row["goal_rgb_sha256"] == root["goal_rgb_sha256"]
                and model_row["history_index"] == root["history_index"], "root identity")
        receipt = model_row["compiler_receipt"]
        require(receipt["strict_selected_index"] is None and len(receipt["proposal_rows"]) == 13,
                "unexpected fixed bank")
        params = np.asarray(model_row["bank_params"], dtype=np.float32)
        raw = np.asarray(model_row["raw_actions"], dtype=np.float32)
        tails = np.asarray(receipt["common_suffix_parameters"], dtype=np.float32)
        require(params.shape == (13, 1, 10) and raw.shape == (13, 5, 2)
                and tails.shape == (5, 2, 10) and np.isfinite(raw).all(), "bad action shapes")
        for i in range(13):
            require(np.allclose(raw[i], decode(params[i], mean, std, lower, upper),
                                rtol=0., atol=1e-6), "prefix decode mismatch")
        tail_raw = [decode(tail, mean, std, lower, upper) for tail in tails]
        require(all(tail.shape == (10, 2) for tail in tail_raw), "tail shape")
        packets, index = setup.sample_history_packets(histories, seed=root["seed"])
        require(index == root["history_index"], "history index mismatch")
        branches = []
        for i in range(13):
            for j in range(5):
                actions = np.ascontiguousarray(np.concatenate((raw[i], tail_raw[j])), dtype=np.float32)
                require(actions.shape == (15, 2) and np.isfinite(actions).all()
                        and bool((np.abs(actions) <= 1.000001).all()), "action outside box")
                branches.append({"prefix_index": i, "tail_index": j, "raw_actions": actions.tolist(),
                                 "actions_sha256": physical.digest_actions(actions)})
        roots.append({**root, "warmup_packets": packets.tolist(),
                      "warmup_sha256": physical.digest_actions(packets), "branches": branches})
    files = (Path(__file__), PROTOCOL, Path(physical.__file__), Path(context_lib.__file__),
             Path(setup.__file__), Path(setup.p0.__file__))
    return {"schema": "pc_response_svd_exhaustive_physical_v1", "source_sha256": SOURCE_SHA,
            "model_plan_sha256": MODEL_PLAN_SHA, "preflight_sha256": setup._sha256_file(preflight_path),
            "sources": {str(p): setup._sha256_file(p) for p in files},
            "history_support": history_receipt, "runtime": physical.runtime_versions(),
            "expected_branches": 130, "roots": roots,
            "scope": "all fixed candidates on two exposed roots; no selected action or P1 admission"}


def run(plan, output, plan_sha):
    import gymnasium as gym

    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "pc_response_svd_physical_events_v1", "plan_sha256": plan_sha,
              "expected_branches": 130, "scope": plan["scope"]}
    journal = (output / "events.jsonl").open("x")

    def emit(value):
        journal.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        journal.flush()
        os.fsync(journal.fileno())

    env = None
    try:
        emit({"kind": "start", **report})
        setup.install_missing_field_compat()
        env = setup.ReacherGoalImageWrapper(
            gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(.65, -.65))
        require(np.array_equal(env.action_space.low, [-1., -1.])
                and np.array_equal(env.action_space.high, [1., 1.]), "action box mismatch")
        for root in plan["roots"]:
            try:
                packets = np.asarray(root["warmup_packets"], dtype=np.float32)
                require(physical.digest_actions(packets) == root["warmup_sha256"], "warmup hash")
                ctx = setup.collect_executed_context(
                    env, seed=root["seed"], history_len=3, action_block=5, warmup_packets=packets,
                    warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
                require(ctx["root_rgb_sha256"] == root["root_rgb_sha256"]
                        and ctx["goal_rgb_sha256"] == root["goal_rgb_sha256"]
                        and ctx["warmup_actions_sha256"] == root["warmup_sha256"], "root identity")
                emit({"kind": "root", "seed": root["seed"], "status": "verified",
                      "root_snapshot": ctx["execution_snapshot"]})
                for branch in root["branches"]:
                    actions = np.asarray(branch["raw_actions"], dtype=np.float32)
                    require(physical.digest_actions(actions) == branch["actions_sha256"], "branch action hash")
                    outcome = physical.execute_branch(env, root["seed"], ctx, actions)
                    emit({"kind": "branch", "seed": root["seed"], **branch, **outcome})
            except Exception as exc:
                emit({"kind": "root_error", "seed": root["seed"], "error_type": type(exc).__name__,
                      "error": str(exc)[:1000]})
        emit({"kind": "finished"})
    finally:
        if env is not None:
            env.close()
        journal.close()


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "source-plan", "preflight", "plan"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = build(args.source, args.source_plan, args.preflight)
    if args.prepare:
        with args.plan.open("x") as stream:
            json.dump(plan, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"plan": str(args.plan), "sha256": setup._sha256_file(args.plan)}))
        return
    require(args.output is not None and args.plan_sha256, "output and frozen plan required")
    require(setup._sha256_file(args.plan) == args.plan_sha256
            and json.loads(args.plan.read_text()) == plan, "physical plan drift")
    run(plan, args.output, args.plan_sha256)
    require(setup._sha256_file(args.plan) == args.plan_sha256
            and json.loads(args.plan.read_text()) == build(args.source, args.source_plan, args.preflight),
            "physical postcheck drift")
    print(json.dumps({"journal_sha256": setup._sha256_file(args.output / "events.jsonl"),
                      "plan_postcheck": True}))


if __name__ == "__main__":
    main()
