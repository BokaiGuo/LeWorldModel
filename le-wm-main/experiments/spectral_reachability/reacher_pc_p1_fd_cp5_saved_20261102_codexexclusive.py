"""CPU-only saved journal consistency checker; not proof of live execution."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import reacher_pc_p1_fd_cp5_host_20261102_codexexclusive as host
from . import reacher_d2p_context as ctx
from . import pc_ap_durable_context_adapter_20261102_codexexclusive as durable


def verify_journal(path):
    events, previous = [], None
    with Path(path).open("rb") as stream:
        for index, line in enumerate(stream):
            host.require(line.endswith(b"\n"), "truncated journal")
            event = json.loads(line)
            digest = event.pop("event_sha256")
            host.require(event["index"] == index and event["previous_sha256"] == previous
                         and durable._digest(event) == digest, "journal chain")
            previous = digest
            events.append(event["payload"])
    host.require(bool(events), "empty journal")
    return events


def verify_context(value, root, arm, role, result=None):
    frames = [ctx.unpack_rgb_observation(x) for x in value["frames"]]
    goal = ctx.unpack_rgb_observation(value["goal"])
    history = np.asarray(value["history"], dtype=np.float32)
    host.require(len(frames) == 3 and history.shape == (3, 10) and np.isfinite(history).all()
                 and ctx.rgb_digest(goal) == root["goal_rgb_sha256"], "context shape or goal")
    warmup = np.asarray(root["warmup_packets"], dtype=np.float32).reshape(3, 10)
    if role == "root":
        host.require(ctx.rgb_digest(frames[-1]) == root["root_rgb_sha256"]
                     and np.array_equal(history, warmup), "root context mismatch")
    elif role == "successor":
        host.require(ctx.rgb_digest(frames[1]) == root["root_rgb_sha256"]
                     and np.array_equal(history[:2], warmup[1:])
                     and np.array_equal(history[2], np.asarray(arm["raw_prefix"], dtype=np.float32).reshape(10)),
                     "successor temporal/action mismatch")
        if result is not None:
            host.require(ctx.rgb_digest(frames[-1]) == result["observed_successor_context"]["root_rgb_sha256"],
                         "successor result mismatch")
    return frames


def consume(plan_path, plan_sha, journal_path, journal_sha):
    host.require(host.bridge.sha_file(plan_path) == plan_sha and
                 host.bridge.sha_file(journal_path) == journal_sha, "external hashes")
    plan = json.loads(Path(plan_path).read_text())
    host.require(plan == host.plan(*plan["inputs"]), "plan or source drift")
    events = verify_journal(journal_path)
    host.require(events[0]["kind"] == "run_started" and events[0]["plan_sha256"] == plan_sha
                 and events[0]["gpu_uuid"] == host.UUID, "run binding")
    roots = plan["bridge"]["roots"]
    roster = [(r["seed"], a["arm"]) for r in roots for a in r["arms"]]
    starts = [(e["seed"], e["arm"]) for e in events if e["kind"] == "arm_started"]
    host.require(len(starts) == len(set(starts)) and all(x in roster for x in starts)
                 and starts == sorted(starts, key=roster.index), "arm roster/order")
    root_order = [e["seed"] for e in events if e["kind"] == "root_started"]
    host.require(root_order == [r["seed"] for r in roots][:len(root_order)], "root order")
    arm_results = []
    for root in roots:
        for arm in root["arms"]:
            matching = [e for e in events if e.get("seed") == root["seed"] and e.get("arm") == arm["arm"]]
            if not matching:
                root_returns = [e for e in events if e["kind"] == "root_returned" and e["seed"] == root["seed"]]
                failures = [a for e in root_returns for a in e["result"]["arms"]
                            if a["arm"] == arm["arm"] and a["status"] == "error"]
                host.require(len(failures) <= 1, "duplicate arm failure")
                arm_results.append({"seed": root["seed"], "arm": arm["arm"],
                                    "status": "error_before_dispatch" if failures else "not_started",
                                    "error": failures[0].get("error") if failures else None})
                continue
            host.require(matching[0]["kind"] == "arm_started", "missing arm intent")
            contexts = [e for e in matching if e["kind"] == "context"]
            roles = [e["role"] for e in contexts]
            host.require(roles == ["root", "successor", "zero_successor"][:len(roles)],
                         "context order")
            returns = [e for e in matching if e["kind"] == "arm_returned"]
            errors = [e for e in matching if e["kind"] == "arm_error"]
            host.require(len(returns) + len(errors) <= 1, "duplicate terminal")
            steps = [e for e in matching if e["kind"] in ("step_intent", "step_returned")]
            intents = [e for e in steps if e["kind"] == "step_intent"]
            returned_steps = [e for e in steps if e["kind"] == "step_returned"]
            host.require([e["index"] for e in intents] == list(range(len(intents)))
                         and [e["index"] for e in returned_steps] == list(range(len(returned_steps)))
                         and len(intents) - len(returned_steps) in (0, 1)
                         and all(e["kind"] == ("step_intent" if i % 2 == 0 else "step_returned")
                                 for i, e in enumerate(steps)), "step lifecycle")
            result = returns[0]["result"] if returns else None
            for event in contexts:
                verify_context(event["value"], root, arm, event["role"], result)
            if result:
                host.require(len(contexts) == 3 and matching[-1]["kind"] == "arm_returned"
                             and np.array_equal(np.asarray(result["first_action_witness_raw"], dtype=np.float32),
                                                np.asarray(arm["raw_prefix"], dtype=np.float32))
                             and result["suffix_receipt"]["forward_rollout_evaluations"] == 32,
                             "result/action/CEM mismatch")
                zero = np.asarray(root["arms"][1]["raw_prefix"], dtype=np.float32)
                host.require(np.array_equal(np.asarray(result["zero_prefix_action_witness_raw"],
                                                       dtype=np.float32), zero),
                             "zero control differs from bank[0] decoded action")
                expected_actions = np.concatenate((np.asarray(arm["raw_prefix"], dtype=np.float32), zero,
                    np.asarray(result["action_witness_raw"], dtype=np.float32)[5:]), axis=0)
                host.require(len(intents) == len(returned_steps) and 10 <= len(intents) <= 20
                             and np.array_equal(np.asarray([e["action"] for e in intents], dtype=np.float32),
                                                expected_actions[:len(intents)]), "physical action trace")
                zero_history = np.asarray(contexts[2]["value"]["history"], dtype=np.float32)
                host.require(np.array_equal(zero_history[:2], np.asarray(root["warmup_packets"],
                                                                       dtype=np.float32).reshape(3, 10)[1:])
                             and np.array_equal(zero_history[-1], zero.reshape(10)),
                             "zero control history differs")
                root_frames = [ctx.unpack_rgb_observation(x) for x in contexts[0]["value"]["frames"]]
                successor_frames = [ctx.unpack_rgb_observation(x) for x in contexts[1]["value"]["frames"]]
                zero_frames = [ctx.unpack_rgb_observation(x) for x in contexts[2]["value"]["frames"]]
                host.require(np.array_equal(root_frames[1], successor_frames[0]) and
                             np.array_equal(root_frames[2], successor_frames[1]) and
                             np.array_equal(root_frames[1], zero_frames[0]) and
                             np.array_equal(root_frames[2], zero_frames[1]) and
                             ctx.rgb_digest(zero_frames[2]) ==
                             result["zero_prefix_successor_context"]["root_rgb_sha256"], "RGB timeline mismatch")
            arm_results.append({"seed": root["seed"], "arm": arm["arm"],
                                "status": "verified_saved" if result else ("error" if errors else "interrupted"),
                                "prefix_execution": result.get("first_physical_execution") if result else None,
                                "zero_control_execution": result.get("zero_prefix_physical_execution") if result else None,
                                "suffix_execution": result.get("suffix_physical_execution") if result else None,
                                "attempted_steps": len(intents), "returned_steps": len(returned_steps),
                                "stop_events": [{"index": e["index"], "terminated": e["terminated"],
                                                 "truncated": e["truncated"]} for e in returned_steps
                                                if e["terminated"] or e["truncated"]],
                                "cem_candidates": 32 if result else None,
                                "other_model_and_physics_cost": "not fully metered"})
    finished = events[-1]["kind"] == "run_finished"
    if finished:
        expected = plan["bridge"]["parameter_hash_expected"]
        host.require(events[-1]["model_before"] == events[-1]["model_after"] == expected,
                     "model hash mismatch")
    return {"status": "complete_saved_consistency" if finished and all(
        a["status"] == "verified_saved" for a in arm_results) else "partial_or_failed",
        "arms": arm_results, "plan_sha256": plan_sha, "journal_sha256": journal_sha,
        "P1_admission": False, "independent_confirmation": False,
        "source_closure_complete": False, "live_execution_proven": False,
        "equal_total_compute": False}


def main():
    parser = argparse.ArgumentParser()
    for arg in ("plan", "plan-sha", "journal", "journal-sha", "output"):
        parser.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    result = consume(args.plan, args.plan_sha, args.journal, args.journal_sha)
    host.write_new(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
