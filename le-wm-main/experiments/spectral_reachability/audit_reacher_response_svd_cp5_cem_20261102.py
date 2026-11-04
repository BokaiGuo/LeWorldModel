"""Validate fixed all-prefix cp5 feedback journal; no simulator replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import reacher_pc_p1_fd_cp5_saved_20261102_codexexclusive as saved
from experiments.spectral_reachability import reacher_pc_p1_fd_cp5_host_20261102_codexexclusive as host
from experiments.spectral_reachability import reacher_response_svd_cp5_cem_20261102 as experiment
from experiments.spectral_reachability import reacher_d2p_context as context_lib


def consume(plan_path, plan_sha, journal_path, journal_sha):
    host.require(host.bridge.sha_file(plan_path) == plan_sha
                 and host.bridge.sha_file(journal_path) == journal_sha, "external hash")
    plan = json.loads(Path(plan_path).read_text())
    host.require(plan == experiment.make_plan(*plan["inputs"]), "plan drift")
    events = saved.verify_journal(journal_path)
    host.require(events[0]["kind"] == "run_started" and events[0]["plan_sha256"] == plan_sha
                 and events[-1]["kind"] == "run_finished"
                 and events[-1]["model_before"] == events[-1]["model_after"] == plan["model_parameter_hash"],
                 "run binding")
    host.require(not any(e["kind"] in ("root_error", "run_error") for e in events), "root or run error")
    host.require([e["seed"] for e in events if e["kind"] == "root_started"]
                 == [r["seed"] for r in plan["roots"]], "root order")
    roster = [(root["seed"], arm["arm"]) for root in plan["roots"] for arm in root["arms"]]
    host.require([(e["seed"], e["arm"]) for e in events if e["kind"] == "arm_started"] == roster,
                 "arm order or missing dispatch")
    results = []
    for root in plan["roots"]:
        zero = np.asarray(root["arms"][0]["raw_prefix"], dtype=np.float32)
        for arm in root["arms"]:
            matching = [e for e in events if e.get("seed") == root["seed"] and e.get("arm") == arm["arm"]]
            host.require(not any(e["kind"] in ("arm_error", "arm_final_error") for e in matching),
                         "arm failed")
            returns = [e for e in matching if e["kind"] == "arm_returned"]
            contexts = [e for e in matching if e["kind"] == "context"]
            host.require(len(returns) == 1 and [e["role"] for e in contexts]
                         == ["root", "successor", "zero_successor"], "incomplete arm")
            result = returns[0]["result"]
            for event in contexts:
                saved.verify_context(event["value"], root, arm, event["role"], result)
            host.require(np.array_equal(np.asarray(result["first_action_witness_raw"], dtype=np.float32),
                                        np.asarray(arm["raw_prefix"], dtype=np.float32))
                         and np.array_equal(np.asarray(result["zero_prefix_action_witness_raw"], dtype=np.float32),
                                            zero)
                         and result["suffix_receipt"]["forward_rollout_evaluations"] == 32, "action or budget")
            zero_history = np.asarray(contexts[2]["value"]["history"], dtype=np.float32)
            warmup = np.asarray(root["warmup_packets"], dtype=np.float32).reshape(3, 10)
            host.require(np.array_equal(zero_history[:2], warmup[1:])
                         and np.array_equal(zero_history[2], zero.reshape(10)), "zero history")
            frames = [[context_lib.unpack_rgb_observation(f) for f in e["value"]["frames"]] for e in contexts]
            host.require(np.array_equal(frames[0][1], frames[1][0])
                         and np.array_equal(frames[0][2], frames[1][1])
                         and np.array_equal(frames[0][1], frames[2][0])
                         and np.array_equal(frames[0][2], frames[2][1])
                         and context_lib.rgb_digest(frames[2][2])
                         == result["zero_prefix_successor_context"]["root_rgb_sha256"], "RGB timeline")
            steps = [e for e in matching if e["kind"] in ("step_intent", "step_returned")]
            intents = [e for e in steps if e["kind"] == "step_intent"]
            returns_step = [e for e in steps if e["kind"] == "step_returned"]
            actions = np.concatenate((np.asarray(arm["raw_prefix"], dtype=np.float32), zero,
                                      np.asarray(result["action_witness_raw"], dtype=np.float32)[5:]))
            host.require(len(intents) == len(returns_step) == 20
                         and all(e["kind"] == ("step_intent" if i % 2 == 0 else "step_returned")
                                 for i, e in enumerate(steps))
                         and [e["index"] for e in intents] == list(range(20))
                         and [e["index"] for e in returns_step] == list(range(20))
                         and not any(e["terminated"] or e["truncated"] for e in returns_step)
                         and np.array_equal(np.asarray([e["action"] for e in intents], dtype=np.float32), actions),
                         "step sequence or stop")
            physical = result["physical"]
            results.append({"seed": root["seed"], "prefix": arm["proposal_index"],
                            "initial": physical["initial_qpos_distance"],
                            "cp5": physical["after_first_chunk_qpos_distance"],
                            "final": physical["final_qpos_distance"],
                            "suffix_raw": result["action_witness_raw"][5:], "cem_candidates": 32})
    return {"status": "complete_saved_consistency", "results": results, "plan_sha256": plan_sha,
            "journal_sha256": journal_sha, "P1_admission": False, "independent_confirmation": False,
            "equal_total_compute": False, "live_execution_proven": False,
            "scope": "26 arms, two exposed roots; no independent physical replay"}


def main():
    parser = argparse.ArgumentParser()
    for key in ("plan", "plan-sha", "journal", "journal-sha", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    result = consume(args.plan, args.plan_sha, args.journal, args.journal_sha)
    host.write_new(args.output, result)
    print(json.dumps({"status": result["status"], "arms": len(result["results"]),
                      "sha256": host.bridge.sha_file(args.output)}))


if __name__ == "__main__":
    main()
