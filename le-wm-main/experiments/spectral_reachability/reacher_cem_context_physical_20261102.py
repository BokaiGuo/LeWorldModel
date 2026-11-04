"""Execute all saved context-sensitivity tails, on exposed development roots."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import reacher_cem_context_sensitivity_saved_20261102 as checked

run = checked.run
host, setup = run.host, run.setup
ROOT = run.ROOT
SENS_PLAN = ROOT / "results/reacher_cem_context_sensitivity_PLAN_20261102_v2.json"
SENS_SHA = "4ec03555ad0506f8a45861e83bb6b9563a902485d307e62771ea9204dba8296a"
SENS_LOG = ROOT / "results/reacher_cem_context_sensitivity_RUN_20261102_v2/events.jsonl"
SENS_LOG_SHA = "77cdf094f5af43a4fa714b6281db868da3fff5acb3856df8b9efdeb684eb3590"
SENS_RECEIPT = ROOT / "results/reacher_cem_context_sensitivity_SAVED_20261102_v1.json"
SENS_RECEIPT_SHA = "548a3f428baee0c7bc146bd8d5a2adddb67a045c1de868b7dbab4d244da810bc"
PROTOCOL = ROOT.parent / "docs/PC_REACHER_CEM_CONTEXT_PHYSICAL_PROTOCOL_20261102.md"
AUTH = "REACHER_CEM_CONTEXT_PHYSICAL_DEVELOPMENT"


def prepare():
    for path, digest in ((SENS_PLAN, SENS_SHA), (SENS_LOG, SENS_LOG_SHA), (SENS_RECEIPT, SENS_RECEIPT_SHA)):
        host.require(setup._sha256_file(path) == digest, "sensitivity input hash")
    plan = json.loads(SENS_PLAN.read_text())
    events = run.saved.verify_journal(SENS_LOG)
    receipt = json.loads(SENS_RECEIPT.read_text())
    recomputed = checked.validate(plan, events, SENS_SHA)
    host.require(plan == run.prepare() and receipt["status"] == recomputed["status"]
                 and receipt["journal_sha256"] == SENS_LOG_SHA and receipt["plan_sha256"] == SENS_SHA
                 and receipt["material_differences"] == recomputed["material_differences"], "source binding")
    old = json.loads(run.physical.OLD_PLAN.read_text())
    stats = Path(old["stats"])
    host.require(setup._sha256_file(stats) == old["files"][str(stats)], "stats hash")
    setup.validate_training_semantic_receipt(stats)
    mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
    lower, upper = setup._normalized_action_bounds(mean, std, np.array([-1., -1.], np.float32),
                                                   np.array([1., 1.], np.float32))
    roots = []
    for root in old["roots"]:
        branches = []
        for pair in (e for e in events if e["kind"] == "pair" and e["seed"] == root["seed"]):
            prefix = np.asarray(root["arms"][pair["prefix"]]["raw_prefix"], np.float32)
            host.require(prefix.shape == (5, 2), "raw prefix shape")
            for label in ("predicted", "observed"):
                params = np.asarray(pair["contexts"][label]["suffix_parameters"], np.float32)
                host.require(params.shape == (2, 10), "tail shape")
                tail = run.physical.old.decode(params, mean, std, lower, upper)
                actions = np.ascontiguousarray(np.concatenate((prefix, tail)), dtype=np.float32)
                host.require(actions.shape == (15, 2) and np.isfinite(actions).all()
                             and np.all(np.abs(actions) <= 1.000001), "action bounds")
                branches.append({"prefix_index": pair["prefix"], "budget": pair["budget"],
                                 "cem_seed": pair["cem_seed"], "context": label,
                                 "actions_sha256": run.physical.physical.digest_actions(actions),
                                 "raw_actions": actions.tolist()})
        host.require(len(branches) == 24, "root denominator")
        roots.append({**{key: root[key] for key in ("seed", "root_rgb_sha256", "goal_rgb_sha256",
                      "history_index", "warmup_packets", "warmup_sha256")}, "branches": branches})
    files = (Path(__file__), PROTOCOL, Path(checked.__file__), Path(run.physical.physical.__file__),
             Path(run.physical.old.__file__), stats)
    return {"schema": "reacher_cem_context_physical_development_v1",
            "sources": {str(p): setup._sha256_file(p) for p in files},
            "sensitivity_plan_sha256": SENS_SHA, "sensitivity_journal_sha256": SENS_LOG_SHA,
            "sensitivity_receipt_sha256": SENS_RECEIPT_SHA, "roots": roots,
            "expected_branches": 48, "new_neural_calls": 0, "independent_confirmation": False,
            "P1_admission": False, "scope": "all saved pairs; exposed roots; offline tail replay"}


def execute(plan_path, plan_sha, output, authority):
    import gymnasium as gym

    host.require(authority == AUTH and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "physical-only authority")
    plan_path, output = Path(plan_path), Path(output)
    host.require(setup._sha256_file(plan_path) == plan_sha and json.loads(plan_path.read_text()) == prepare(),
                 "plan/source drift")
    host.require(output.is_absolute() and output.parent == ROOT / "results"
                 and output.name.startswith("reacher_cem_context_physical_RUN_")
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
            try:
                packets = np.asarray(root["warmup_packets"], np.float32)
                host.require(run.physical.physical.digest_actions(packets) == root["warmup_sha256"], "warmup")
                ctx = setup.collect_executed_context(
                    env, seed=root["seed"], history_len=3, action_block=5, warmup_packets=packets,
                    warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
                host.require(ctx["root_rgb_sha256"] == root["root_rgb_sha256"]
                             and ctx["goal_rgb_sha256"] == root["goal_rgb_sha256"]
                             and ctx["warmup_actions_sha256"] == root["warmup_sha256"], "root identity")
                journal.emit({"kind": "root_verified", "seed": root["seed"]})
                for index, branch in enumerate(root["branches"]):
                    actions = np.asarray(branch["raw_actions"], np.float32)
                    host.require(run.physical.physical.digest_actions(actions) == branch["actions_sha256"], "actions")
                    journal.emit({"kind": "branch_started", "seed": root["seed"], "branch_index": index,
                                  "actions_sha256": branch["actions_sha256"]})
                    original_step, counter = env.step, [0]

                    def step(action):
                        i = counter[0]
                        host.require(i < 15 and np.array_equal(np.asarray(action, np.float32), actions[i]),
                                     "step action mismatch")
                        journal.emit({"kind": "step_intent", "seed": root["seed"], "branch_index": index,
                                      "step": i, "action": np.asarray(action, np.float32).tolist()})
                        result = original_step(action)
                        journal.emit({"kind": "step_returned", "seed": root["seed"], "branch_index": index,
                                      "step": i, "terminated": bool(result[2]), "truncated": bool(result[3])})
                        counter[0] += 1
                        return result
                    try:
                        env.step = step
                        outcome = run.physical.physical.execute_branch(env, root["seed"], ctx, actions)
                    finally:
                        env.step = original_step
                    journal.emit({"kind": "branch_returned", "seed": root["seed"], "branch_index": index,
                                  "outcome": outcome})
            except Exception as exc:
                journal.emit({"kind": "root_error", "seed": root["seed"], "type": type(exc).__name__,
                              "error": str(exc)[:1000]})
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
