"""Replay all sealed development action pairs with full RGB trajectories."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import reacher_rgb_direction_same_action_20261102 as source

host, setup, ROOT = source.host, source.setup, source.ROOT
pair = source.paired
physical = source.sensitivity.physical.physical
context = pair.audit.context_lib
PROTOCOL = ROOT.parent / "docs/PC_REACHER_PAIR_REPLAY_PROTOCOL_20261102.md"
AUTH = "REACHER_FIXED_PAIR_RGB_REPLAY_DEVELOPMENT"


def prepare():
    source.prepare()
    plan = json.loads(source.PLAN.read_text())
    event_sets = [pair.audit.saved.verify_journal(p) for p in (pair.prior.L2_JOURNAL, source.JOURNAL)]
    roots = []
    for root in plan["roots"]:
        branches = []
        for arm in root["arms"]:
            records = []
            for label, events in zip(("l2", "direction"), event_sets, strict=True):
                found = [e for e in events if e.get("seed") == root["seed"]
                         and e.get("arm") == arm["arm"] and e["kind"] == "arm_returned"]
                contexts = [e for e in events if e.get("seed") == root["seed"]
                            and e.get("arm") == arm["arm"] and e["kind"] == "context"
                            and e["role"] == "successor"]
                host.require(len(found) == len(contexts) == 1, "source branch/context")
                result, ctx = found[0]["result"], contexts[0]["value"]
                actions = np.asarray(result["action_witness_raw"], dtype=np.float32)
                host.require(actions.shape == (15, 2) and np.isfinite(actions).all()
                             and np.max(np.abs(actions)) <= 1.000001, "action witness")
                host.require(np.array_equal(actions[:5], np.asarray(arm["raw_prefix"], np.float32)),
                             "prefix binding")
                cp5 = context.rgb_digest(context.unpack_rgb_observation(ctx["frames"][-1]))
                records.append({"label": label, "prefix": arm["proposal_index"], "arm": arm["arm"],
                                "actions": actions.tolist(),
                                "actions_sha256": physical.digest_actions(actions),
                                "cp5_rgb_sha256": cp5,
                                "source_final_distance": result["physical"]["final_qpos_distance"]})
            host.require(records[0]["cp5_rgb_sha256"] == records[1]["cp5_rgb_sha256"], "paired cp5")
            branches.extend(records)
        roots.append({**{k: root[k] for k in ("seed", "root_rgb_sha256", "goal_rgb_sha256",
                      "history_index", "warmup_packets", "warmup_sha256")}, "branches": branches})
    host.require(len(roots) == 2 and all(len(r["branches"]) == 26 for r in roots), "roster")
    files = [Path(__file__), PROTOCOL, Path(physical.__file__), Path(context.__file__)]
    return {"schema": "reacher_fixed_pair_rgb_replay_v1", "roots": roots, "expected_branches": 52,
            "files": {str(p): setup._sha256_file(p) for p in files},
            "source_plan_sha256": source.PLAN_SHA, "source_journals": [pair.prior.L2_JOURNAL_SHA,
            source.JOURNAL_SHA], "source_receipt_sha256": source.RECEIPT_SHA,
            "distance_tolerance": 1e-6, "P1_admission": False}


def execute(plan_path, plan_sha, output, authority):
    import gymnasium as gym

    host.require(authority == AUTH and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "physical authority")
    plan_path, output = Path(plan_path), Path(output)
    host.require(setup._sha256_file(plan_path) == plan_sha and json.loads(plan_path.read_text()) == prepare(),
                 "plan/source drift")
    host.require(output.is_absolute() and output.parent == ROOT / "results"
                 and output.name.startswith("reacher_fixed_pair_rgb_replay_RUN_")
                 and not os.path.lexists(output), "exclusive output")
    output.mkdir(mode=0o700)
    journal = host.durable.Journal(output / "events.jsonl")
    env = None
    try:
        journal.emit({"kind": "start", "plan_sha256": plan_sha})
        setup.install_missing_field_compat()
        env = setup.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
                                             target_delta=(.65, -.65))
        host.require(np.array_equal(env.action_space.low, [-1., -1.])
                     and np.array_equal(env.action_space.high, [1., 1.]), "action bounds")
        for root in json.loads(plan_path.read_text())["roots"]:
            packets = np.asarray(root["warmup_packets"], np.float32)
            host.require(physical.digest_actions(packets) == root["warmup_sha256"], "warmup hash")
            ctx = setup.collect_executed_context(
                env, seed=root["seed"], history_len=3, action_block=5, warmup_packets=packets,
                warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
            host.require(ctx["root_rgb_sha256"] == root["root_rgb_sha256"]
                         and ctx["goal_rgb_sha256"] == root["goal_rgb_sha256"]
                         and ctx["warmup_actions_sha256"] == root["warmup_sha256"], "root identity")
            journal.emit({"kind": "root_verified", "seed": root["seed"]})
            for index, branch in enumerate(root["branches"]):
                actions = np.asarray(branch["actions"], np.float32)
                host.require(physical.digest_actions(actions) == branch["actions_sha256"], "action hash")
                journal.emit({"kind": "branch_started", "seed": root["seed"], "index": index,
                              "actions_sha256": branch["actions_sha256"]})
                original, counter = env.step, [0]

                def step(action):
                    i = counter[0]
                    host.require(i < 15 and np.array_equal(np.asarray(action, np.float32), actions[i]),
                                 "step action")
                    journal.emit({"kind": "step_intent", "seed": root["seed"], "index": index,
                                  "step": i, "action": actions[i].tolist()})
                    result = original(action)
                    journal.emit({"kind": "step_returned", "seed": root["seed"], "index": index,
                                  "step": i, "terminated": bool(result[2]), "truncated": bool(result[3])})
                    counter[0] += 1
                    return result

                try:
                    env.step = step
                    result = physical.execute_branch(env, root["seed"], ctx, actions)
                finally:
                    env.step = original
                journal.emit({"kind": "branch_returned", "seed": root["seed"], "index": index,
                              "outcome": result})
                host.require(result["status"] == "complete" and result["returned_steps"] == 15,
                             "incomplete replay, retained in journal")
                cp5 = context.rgb_digest(context.unpack_rgb_observation(result["trajectory"][5]["rgb"]))
                final = result["trajectory"][15]["qpos_distance"]
                host.require(cp5 == branch["cp5_rgb_sha256"]
                             and abs(final - branch["source_final_distance"]) <= 1e-6,
                             "source cp5 or final distance mismatch")
            print(json.dumps({"seed": root["seed"], "branches": len(root["branches"])}), flush=True)
        host.require(json.loads(plan_path.read_text()) == prepare(), "postcheck drift")
        journal.emit({"kind": "finished", "plan_sha256": plan_sha})
    finally:
        if env is not None:
            env.close()
        journal.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    for name in ("plan", "sha256", "output", "authority"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    if args.command == "prepare":
        host.write_new(args.plan, prepare())
        print(json.dumps({"sha256": setup._sha256_file(Path(args.plan))}))
    else:
        execute(args.plan, args.sha256, args.output, args.authority)
        print(json.dumps({"journal_sha256": setup._sha256_file(Path(args.output) / "events.jsonl")}))


if __name__ == "__main__":
    main()
