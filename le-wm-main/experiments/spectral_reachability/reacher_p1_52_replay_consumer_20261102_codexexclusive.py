"""Saved-only replay journal consumer. It does not rerun physics or inference."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import reacher_p1_52_replay_contract_20261102_codexexclusive as contract
from . import reacher_p1_52_replay_roster_20261102_codexexclusive as roster
from . import reacher_pc_p1_fd_cp5_host_20261102_codexexclusive as host
from . import reacher_pc_p1_fd_cp5_saved_20261102_codexexclusive as saved
from . import reacher_d2p_context as ctx


def consume_events(events, rows, plan_sha, model_hash, roots, action_mean, action_std):
    host.require(events and events[0] == {"kind": "run_started", "plan_sha256": plan_sha},
                 "run start binding")
    keys = [(r["group"], r["seed"], r["arm"]) for r in rows]
    roster.validate_roster(rows)
    started = [e for e in events if e["kind"] == "arm_started"]
    host.require([(e["group"], e["seed"], e["arm"]) for e in started] == keys[:len(started)],
                 "arm order")
    arm_key_events = [(e.get("group"), e.get("seed"), e.get("arm")) for e in events if "arm" in e]
    host.require(arm_key_events == sorted(arm_key_events, key=keys.index), "interleaved arms")
    warmup_by_seed = {r["seed"]: np.asarray(r["warmup_packets"], np.float32).reshape(15, 2) for r in roots}
    host.require(not any(e["kind"] == "run_finished" for e in events[:-1]), "run final order")
    finished = events[-1]["kind"] == "run_finished"
    if finished:
        host.require(len(started) == 52 and events[-1]["model_before"] == model_hash
                     and events[-1]["model_after"] == model_hash, "run postcheck")
    results = []
    for row in rows:
        key = (row["group"], row["seed"], row["arm"])
        arm_events = [e for e in events if (e.get("group"), e.get("seed"), e.get("arm")) == key]
        base = {"group": key[0], "seed": key[1], "arm": key[2], "prefix_index": row["prefix_index"]}
        if not arm_events:
            results.append({**base, "status": "not_started"})
            continue
        host.require(arm_events[0]["kind"] == "arm_started"
                     and arm_events[0]["action_sha256"] == row["action_sha256"], "arm start")
        terminals = [e for e in arm_events if e["kind"] in ("arm_complete", "arm_failed")]
        host.require(len(terminals) <= 1 and (not terminals or terminals[0] is arm_events[-1]),
                     "arm termination")
        steps = [e for e in arm_events if e["kind"] in ("step_intent", "step_returned")]
        for n, event in enumerate(steps):
            host.require(event["kind"] == ("step_intent" if n % 2 == 0 else "step_returned")
                         and event["step"] == n // 2 and n // 2 < 15, "step chronology")
            if n % 2 == 0:
                host.require(np.array_equal(np.asarray(event["action"], dtype=np.float32),
                                            np.asarray(row["actions"][n // 2], dtype=np.float32)), "step action")
        stopped = [n for n, e in enumerate(steps) if e["kind"] == "step_returned"
                   and (e["terminated"] or e["truncated"])]
        host.require(not stopped or stopped == [len(steps) - 1], "step after termination")
        if not terminals or terminals[0]["kind"] == "arm_failed":
            if terminals:
                host.require(terminals[0]["returned_steps"] == len(steps) // 2, "failed step count")
            results.append({**base, "status": "failed" if terminals else "interrupted",
                            "failure": terminals[0] if terminals else None,
                            "step_intents": (len(steps)+1)//2, "step_returns": len(steps)//2})
            continue
        warmup = [e for e in arm_events if e["kind"] in ("warmup_intent", "warmup_returned")]
        host.require(len(warmup) == 30 and all(
            e["kind"] == ("warmup_intent" if n % 2 == 0 else "warmup_returned") and e["step"] == n // 2
            for n, e in enumerate(warmup)), "warmup lifecycle")
        host.require(all(np.array_equal(np.asarray(e["action"], np.float32), warmup_by_seed[row["seed"]][i])
                         for i, e in enumerate(warmup[::2])), "warmup action binding")
        host.require(all(not e["terminated"] and not e["truncated"] for e in warmup[1::2] + steps[1::2]),
                     "completed arm stopped")
        kinds = [e["kind"] + (":" + e["phase"] if "phase" in e else "") for e in arm_events
                 if e["kind"] not in ("step_intent", "step_returned", "warmup_intent", "warmup_returned")]
        host.require(kinds == ["arm_started", "cp5_verified", "model_intent:prediction",
                               "model_returned:prediction", "physical_returned",
                               "model_intent:actual_rgb_encode", "model_returned:actual_rgb_encode",
                               "arm_complete"] and len(steps) == 30, "complete event lifecycle")
        host.require(arm_events.index(warmup[-1]) < arm_events.index(steps[0])
                     and arm_events.index(steps[9]) < next(i for i, e in enumerate(arm_events)
                                                         if e["kind"] == "cp5_verified")
                     and next(i for i, e in enumerate(arm_events) if e["kind"] == "model_returned"
                              and e.get("phase") == "prediction") < arm_events.index(steps[10])
                     and arm_events.index(steps[-1]) < next(i for i, e in enumerate(arm_events)
                                                             if e["kind"] == "physical_returned"),
                     "interleaved model/physics chronology")
        cp5 = next(e for e in arm_events if e["kind"] == "cp5_verified")
        physical = next(e for e in arm_events if e["kind"] == "physical_returned")
        prediction = next(e for e in arm_events if e["kind"] == "model_returned" and e["phase"] == "prediction")
        intent = next(e for e in arm_events if e["kind"] == "model_intent" and e["phase"] == "prediction")
        actual = next(e for e in arm_events if e["kind"] == "model_returned" and e["phase"] == "actual_rgb_encode")
        host.require(roster.digest(cp5["context"]) == row["cp5_context_sha256"], "cp5 context binding")
        normalized = (np.asarray(row["actions"][5:], np.float32).reshape(2, 10)
                      - np.asarray(action_mean, np.float32)) / np.asarray(action_std, np.float32)
        host.require(intent["action_sha256"] == row["action_sha256"]
                     and np.array_equal(np.asarray(intent["suffix_normalized"], np.float32), normalized),
                     "prediction action token mismatch")
        final = physical["context"]
        frames = [ctx.unpack_rgb_observation(x) for x in final["frames"]]
        goal = ctx.unpack_rgb_observation(final["goal"])
        old_frames = [ctx.unpack_rgb_observation(x) for x in cp5["context"]["frames"]]
        host.require(len(frames) == 3 and np.array_equal(frames[0], old_frames[-1])
                     and ctx.rgb_digest(goal) == row["goal_rgb_sha256"]
                     and np.array_equal(np.asarray(final["history"], np.float32)[-2:],
                                        np.asarray(row["actions"][5:], np.float32).reshape(2, 10)),
                     "terminal RGB/history binding")
        value = physical["physical_distance"]
        host.require(abs(cp5["physical_distance"] - row["original_physical"]["after_first_chunk_qpos_distance"]) <= 1e-8
                     and abs(value - row["original_physical"]["final_qpos_distance"]) <= 1e-8
                     and physical["original_final"] == row["original_physical"]["final_qpos_distance"],
                     "physical replay mismatch")
        recomputed = contract.metrics(prediction["predicted"], actual["actual"], prediction["goal"], value)
        host.require(all(math.isclose(recomputed[k], arm_events[-1]["metrics"][k], rel_tol=1e-12, abs_tol=1e-12)
                         for k in recomputed), "metric mismatch")
        results.append({**base, "status": "complete", **recomputed,
                        "terminal_rgb_sha256": ctx.rgb_digest(frames[-1])})
    ranked = []
    for start in range(0, 52, 13):
        ranked.extend(roster.rank_complete_panel(results[start:start + 13]))
    return {"status": "complete" if finished and all(r["status"] == "complete" for r in results) else "partial",
            "rows": ranked, "completed": sum(r["status"] == "complete" for r in results),
            "run_terminal": events[-1]["kind"], "independent_outcome": False,
            "P1_admission": False, "equal_total_compute": False,
            "scope": "saved consistency only; not independent proof of physics or neural operations"}


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan", "sha256", "journal", "journal-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    plan = contract.bind_plan(args.plan, args.sha256)
    host.require(contract.sha(args.journal) == args.journal_sha256, "journal hash")
    rows = json.loads(Path(plan["roster"]).read_text())["rows"]
    stats = json.loads(Path(plan["stats"]).read_text())
    matches = [r for r in stats["reports"] if r["environment"] == "swm/ReacherDMControl-v0"]
    host.require(len(matches) == 1, "action stats")
    result = consume_events(saved.verify_journal(args.journal), rows, args.sha256,
                            plan["model_parameter_hash"], plan["roots"],
                            matches[0]["calibration_action_mean"], matches[0]["calibration_action_std"])
    result["journal_sha256"] = args.journal_sha256
    host.write_new(args.output, result)
    print(json.dumps({"status": result["status"], "completed": result["completed"],
                      "sha256": contract.sha(args.output)}))


if __name__ == "__main__":
    main()
