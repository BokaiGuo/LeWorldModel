"""Execute four previously selected predicted-tail action programs, no neural calls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import reacher_response_svd_predicted_cem_20261102 as compiler
from . import reacher_response_svd_predicted_cem_saved_20261102 as checked
from . import run_reacher_pc_response_svd_physical_20261102 as old
from . import run_reacher_pc_p1_fd_common_physical_20261102_codexexclusive as physical

ROOT = compiler.ROOT
host = compiler.host
MODEL_PLAN = ROOT / "results/reacher_response_svd_predicted_cem_PLAN_20261102_v2.json"
MODEL_PLAN_SHA = "7d7e3160b7154f2c01301bdc36c10a3c764da8ebc074482d3b8a4320bd17aa95"
REPORT = ROOT / "results/reacher_response_svd_predicted_cem_RUN_20261102_v1/report.json"
REPORT_SHA = "489f72132db81469ff02dd40f95474b188e3845b87f3070923920dd0aabf0c06"
RECEIPT = ROOT / "results/reacher_response_svd_predicted_cem_SAVED_20261102_v1.json"
RECEIPT_SHA = "d47604072fd5e113336af9e4ae156545f4e7426eafbf887089d76e7ac970dd21"
OLD_PLAN = ROOT / "results/reacher_response_svd_cp5_PLAN_20261102_v1.json"
OLD_PLAN_SHA = "f1087157d9cb4f6d4efdfc9900c74fed66a28f33169835e61093033c79077014"
PROTOCOL = ROOT.parent / "docs/PC_REACHER_RESPONSE_SVD_PREDICTED_CEM_PHYSICAL_PROTOCOL_20261102.md"
AUTH = "RESPONSE_SVD_PREDICTED_TAIL_KNOWN_ROOT_PHYSICAL"


def make_plan():
    for path, digest in ((MODEL_PLAN, MODEL_PLAN_SHA), (REPORT, REPORT_SHA),
                         (RECEIPT, RECEIPT_SHA), (OLD_PLAN, OLD_PLAN_SHA),
                         (checked.PHYSICAL, checked.PHYSICAL_SHA), (checked.FEEDBACK, checked.FEEDBACK_SHA)):
        host.require(compiler.setup._sha256_file(path) == digest, "sealed input hash")
    model_plan, report, receipt, old_plan = [json.loads(p.read_text()) for p in
                                               (MODEL_PLAN, REPORT, RECEIPT, OLD_PLAN)]
    host.require(model_plan == compiler.make_plan() and report["plan_sha256"] == MODEL_PLAN_SHA
                 and receipt["report_sha256"] == REPORT_SHA and receipt["status"] == "saved_selection_recomputed",
                 "selection source binding")
    checked.validate(report, model_plan, json.loads(compiler.SOURCE.read_text()),
                     json.loads(checked.PHYSICAL.read_text()), json.loads(checked.FEEDBACK.read_text()))
    stats = Path(old_plan["stats"])
    host.require(compiler.setup._sha256_file(stats) == old_plan["files"][str(stats)], "stats hash")
    compiler.setup.validate_training_semantic_receipt(stats)
    mean, std = compiler.setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
    lower, upper = compiler.setup._normalized_action_bounds(
        mean, std, np.array([-1., -1.], np.float32), np.array([1., 1.], np.float32))
    roots = []
    for source, root, recorded in zip(report["rows"], old_plan["roots"], receipt["rows"], strict=True):
        host.require(source["seed"] == root["seed"] == recorded["seed"], "root order")
        selected = source["selection"]["selected_proposal_index"]
        host.require(selected is not None and selected == recorded["selected"] and selected != 0, "selected prefix")
        branches = []
        for index, role in ((selected, "selected"), (0, "neutral")):
            prefix = np.asarray(root["arms"][index]["params"], np.float32)
            decoded_prefix = old.decode(prefix, mean, std, lower, upper)
            raw_prefix = np.asarray(root["arms"][index]["raw_prefix"], np.float32)
            host.require(np.allclose(decoded_prefix, raw_prefix, rtol=0, atol=1e-6),
                         "decoded prefix versus physical receipt")
            cpu_decode_max_error = float(np.max(np.abs(decoded_prefix - raw_prefix)))
            tail = np.asarray(source["selection"]["proposal_rows"][index]["suffix_parameters"], np.float32)
            host.require(tail.shape == (2, 10), "suffix shape")
            raw_tail = old.decode(tail, mean, std, lower, upper)
            raw = np.ascontiguousarray(np.concatenate([raw_prefix, raw_tail]), dtype=np.float32)
            host.require(raw.shape == (15, 2) and np.isfinite(raw).all() and (np.abs(raw) <= 1.000001).all(),
                         "bounded action chain")
            branches.append({"role": role, "prefix_index": index, "raw_actions": raw.tolist(),
                             "prefix_cpu_decode_max_abs_error": cpu_decode_max_error,
                             "actions_sha256": physical.digest_actions(raw)})
        roots.append({**{k: root[k] for k in ("seed", "root_rgb_sha256", "goal_rgb_sha256", "history_index",
                                               "warmup_packets", "warmup_sha256")}, "branches": branches})
    files = (Path(__file__), PROTOCOL, Path(old.__file__), Path(physical.__file__),
             Path(compiler.__file__), Path(checked.__file__), Path(compiler.setup.__file__),
             Path(compiler.setup.p0.__file__), stats)
    return {"schema": "response_svd_predicted_tail_physical_v1",
            "sources": {str(p): compiler.setup._sha256_file(p) for p in files},
            "model_plan_sha256": MODEL_PLAN_SHA, "model_report_sha256": REPORT_SHA,
            "selection_receipt_sha256": RECEIPT_SHA, "previous_plan_sha256": OLD_PLAN_SHA,
            "roots": roots, "expected_branches": 4, "new_neural_calls": 0,
            "independent_outcome": False, "P1_admission": False,
            "scope": "saved model-selected prefixes and their predicted tails; exposed roots; open loop"}


def run(plan_path, expected, output, authority):
    import gymnasium as gym

    host.require(authority == AUTH and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "physical-only authority")
    plan_path = Path(plan_path)
    host.require(compiler.setup._sha256_file(plan_path) == expected
                 and json.loads(plan_path.read_text()) == make_plan(), "plan/source drift")
    output = Path(output)
    host.require(output.is_absolute() and output.parent == ROOT / "results"
                 and output.name.startswith("reacher_response_svd_predicted_tail_RUN_")
                 and not os.path.lexists(output), "exclusive output")
    output.mkdir(mode=0o700)
    journal = host.durable.Journal(output / "events.jsonl")
    env = None
    try:
        journal.emit({"kind": "start", "plan_sha256": expected})
        compiler.setup.install_missing_field_compat()
        env = compiler.setup.ReacherGoalImageWrapper(
            gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(.65, -.65))
        host.require(np.array_equal(env.action_space.low, [-1., -1.])
                     and np.array_equal(env.action_space.high, [1., 1.]), "action bounds")
        for root in make_plan()["roots"]:
            try:
                packets = np.asarray(root["warmup_packets"], np.float32)
                host.require(physical.digest_actions(packets) == root["warmup_sha256"], "warmup hash")
                ctx = compiler.setup.collect_executed_context(
                    env, seed=root["seed"], history_len=3, action_block=5, warmup_packets=packets,
                    warmup_policy=compiler.setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
                host.require(ctx["root_rgb_sha256"] == root["root_rgb_sha256"]
                             and ctx["goal_rgb_sha256"] == root["goal_rgb_sha256"]
                             and ctx["warmup_actions_sha256"] == root["warmup_sha256"], "root identity")
                journal.emit({"kind": "root_verified", "seed": root["seed"]})
                for branch in root["branches"]:
                    actions = np.asarray(branch["raw_actions"], np.float32)
                    host.require(physical.digest_actions(actions) == branch["actions_sha256"], "action hash")
                    journal.emit({"kind": "branch_started", "seed": root["seed"], "role": branch["role"],
                                  "actions_sha256": branch["actions_sha256"]})
                    outcome = physical.execute_branch(env, root["seed"], ctx, actions)
                    journal.emit({"kind": "branch_returned", "seed": root["seed"], **branch, "outcome": outcome})
            except Exception as exc:
                journal.emit({"kind": "root_error", "seed": root["seed"], "type": type(exc).__name__,
                              "error": str(exc)[:1000]})
        host.require(json.loads(plan_path.read_text()) == make_plan(), "postcheck drift")
        journal.emit({"kind": "finished", "plan_sha256": expected})
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
        host.write_new(args.plan, make_plan())
        print(json.dumps({"sha256": compiler.setup._sha256_file(Path(args.plan))}))
    else:
        run(args.plan, args.sha256, args.output, args.authority)
        print(json.dumps({"journal_sha256": compiler.setup._sha256_file(Path(args.output) / "events.jsonl")}))


if __name__ == "__main__":
    main()
