"""Saved-only audit of the four-trajectory, three-prefix exploration.

This deliberately does not rebuild or open H10 sources or outputs.  The pinned
intervention prepare is used as the schedule authority, not as proof that its
upstream source closure is independently authentic.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

NAME = "codex_four_prefix_saved_audit_20261030"
ARMS = ("structured", "native", "random", "immediate_progress")
PREFIXES = ("selected", "zero", "greedy")
SUFFIXES = ("saved", "zero")
PREP_SHA = "9dc528844fede250ac6eacf008dc1b3b7b5da9545e1d97fa17a78f4ea0e98dfb"
REPORT_SHA = "d8086e1da759544af775974f0b161d5de432cca5fbc445109727f1e73ab2bdf7"


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "regular input")
    return json.loads(path.read_bytes())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def journal(path, tail):
    events = []
    previous = None
    with Path(path).open("rb") as stream:
        for i, line in enumerate(stream):
            need(line.endswith(b"\n"), "truncated journal")
            event = json.loads(line)
            need(event["index"] == i and event["previous_sha256"] == previous
                 and event["event_sha256"] == digest({k: v for k, v in event.items()
                                                       if k != "event_sha256"}), "journal chain")
            previous = event["event_sha256"]
            events.append(event)
    need(events and previous == tail, "journal tail")
    return events


def schedule_check(schedule):
    need(len(schedule) == 24, "schedule denominator")
    for i, row in enumerate(schedule):
        need((row["index"], row["trajectory_arm"], row["prefix_role"], row["suffix_role"])
             == (i, ARMS[i // 6], PREFIXES[(i % 6) // 2], SUFFIXES[i % 2]), "schedule identity")
        a = np.asarray(row["commands"])
        need(a.shape == (15, 2) and np.isfinite(a).all() and (np.abs(a) <= 1).all()
             and np.array_equal(a, a.astype(np.float32))
             and row["commands"] == row["prefix"] + row["suffix"], "action schedule")
        if row["prefix_role"] == "zero":
            need(not np.any(a[:5]), "zero prefix")
        if row["suffix_role"] == "zero":
            need(not np.any(a[5:]), "zero suffix")
    for j in range(4):
        block = schedule[6*j:6*j+6]
        need(all(x["suffix"] == block[0]["suffix"] for x in block if x["suffix_role"] == "saved"),
             "shared saved suffix")
        for k in range(3):
            need(block[2*k]["prefix"] == block[2*k+1]["prefix"], "paired prefix")
    need(all(x["prefix"] == schedule[4]["prefix"] for x in schedule if x["prefix_role"] == "greedy"),
         "shared greedy")


def audit(directory, prepared):
    directory, prepared = Path(directory), Path(prepared)
    need(directory.is_absolute() and directory.resolve() == directory and directory.is_dir(), "run directory")
    need(sha(directory / "report.json") == REPORT_SHA and sha(prepared) == PREP_SHA, "pinned inputs")
    report, prep = read(directory / "report.json"), read(prepared)
    need(report["schema"] == "codex_four_arm_prefix_intervention_20261030.report.v1"
         and report["status"] == "recorded_not_independently_consumed"
         and report["fixed_branches"] == len(report["rows"]) == 24
         and report["P1_admission"] is False and report["independent_confirmation"] is False
         and report["fairness_proven"] is False and not report["errors"]
         and not report["pending"] and not report["secondary_errors"], "complete report")
    need(report["prepared"] == str(prepared) and report["prepare_sha256"] == PREP_SHA
         and read(directory / "execution_binding.json") == {"prepared": str(prepared),
                                                               "prepare_sha256": PREP_SHA},
         "prepare binding")
    schedule = prep["schedule"]
    need(prep["fixed_branches"] == 24 and prep["P1_admission"] is False, "prepare contract")
    schedule_check(schedule)
    roster = read(directory / "roster.json")
    need(roster["schedule"] == schedule and len(roster["rows"]) == 24, "saved roster")
    jp = directory / "journal.jsonl"
    need(sha(jp) == report["journal_sha256"], "journal SHA")
    events = journal(jp, report["journal_chain_tail"])
    need(events[0]["kind"] == "run_started" and events[0]["fixed_branches"] == 24
         and events[0]["caps"] == prep["caps"] and events[-1]["kind"] == "run_finished"
         and events[-1]["status"] == report["status"] and events[-1]["counts"] == report["counts"],
         "journal boundaries")
    allowed = {"run_started", "run_finished", "operation_attempted", "operation_completed",
               "fresh_observation", "guard_passed", "warmup_return", "branch_dispatch",
               "branch_returned", "branch_closed", "censored_out_of_domain"}
    need(all(e["kind"] in allowed for e in events), "unexpected journal event")
    pending, counts, by_episode = {}, Counter(), defaultdict(list)
    branch_events = defaultdict(list)
    next_op = 0
    for e in events:
        if "episode_id" in e:
            by_episode[e["episode_id"]].append(e)
        if "branch" in e:
            branch_events[e["branch"]].append(e)
        if e["kind"] == "operation_attempted":
            need(e["operation_id"] == next_op and e["units"] == 1, "operation sequence")
            next_op += 1
            pending[e["operation_id"]] = e
            counts[e["resource"]] += 1
        if e["kind"] == "operation_completed":
            a = pending.pop(e["operation_id"], None)
            need(a is not None and all(a[k] == e[k] for k in ("resource", "units", "episode_id", "region")),
                 "operation completion")
    need(not pending and set(counts) == set(prep["caps"]), "complete operations")
    for kind, cap in prep["caps"].items():
        need(counts[kind] <= cap and report["counts"][kind] ==
             {"attempted": counts[kind], "completed": counts[kind]}, "operation census")
    closed = [e for e in events if e["kind"] == "branch_closed"]
    need([e["branch"] for e in closed] == list(range(24)), "ordered branch closure")
    summaries = []
    for i, (plan, row, close) in enumerate(zip(schedule, report["rows"], closed, strict=True)):
        episode = f"branch_{i:02d}"
        need(row["episode_id"] == episode and row["branch"] == i
             and all(row[k] == plan[k] for k in ("trajectory_arm", "prefix_role", "suffix_role"))
             and row["status"] == close["status"] and row["first_error"] is None
             and not row["secondary_errors"] and read(directory / (episode + "_result.json")) == row,
             "branch report identity")
        scope = by_episode[episode]
        frames = [e for e in scope if e["kind"] == "fresh_observation"]
        guards = [e for e in scope if e["kind"] == "guard_passed"]
        returned = [e for e in branch_events[i] if e["kind"] == "branch_returned"]
        dispatched = [e for e in branch_events[i] if e["kind"] == "branch_dispatch"]
        warm = [e for e in scope if e["kind"] == "warmup_return"]
        local_counts = Counter(e["resource"] for e in scope if e["kind"] == "operation_attempted")
        need(all(row["physical_counts"][k] == {"attempted": local_counts[k], "completed": local_counts[k]}
                 for k in prep["caps"]), "branch operation census")
        n = row["control_returned"]
        need(len(returned) == len(dispatched) == len(row["transitions"]) == len(row["executed_raw"]) == n
             and len(warm) == row["warmup_returned"] == 10 and row["initial_exact"] is True, "action census")
        need([e["step"] for e in warm] == list(range(1, 11))
             and all(not e["terminated"] and not e["truncated"] for e in warm)
             and [e["command"] for e in warm] ==
             [[float(np.float32(.1)), 0.0]] * 5 + [[0.0, float(np.float32(.1))]] * 5, "warmup returns")
        for j, (d, r, t, cmd) in enumerate(zip(dispatched, returned, row["transitions"],
                                                plan["commands"][:n], strict=True), 1):
            need(d["step"] == r["step"] == t["step"] == j and d["command"] == r["command"] == cmd
                 and row["executed_raw"][j-1] == cmd and d["index"] < r["index"] < close["index"]
                 and all(r[k] == t[k] for k in ("branch", "step", "command", "state", "n_contacts",
                                                "terminated", "truncated"))
                 and type(r["n_contacts"]) is int and r["n_contacts"] >= 0
                 and len(r["state"]) == 7 and all(math.isfinite(x) for x in r["state"]),
                 "literal action and returned state")
        need(all(not (r["terminated"] or r["truncated"]) for r in returned[:-1]), "post-stop action")
        need(len(frames) == row["PNG"] == 11 + 1 + n, "frame count")
        expected = [("warmup", j) for j in range(11)] + [("plan", j) for j in range(n+1)]
        need([(e["receipt"]["phase"], e["receipt"]["step"]) for e in frames] == expected, "frame schedule")
        need(all(e["receipt"]["goal"] == frames[11]["receipt"]["goal"]
                 for e in frames if e["receipt"]["phase"] == "plan"), "plan goal drift")
        censor = row["status"] == "censored_out_of_domain"
        need(len(guards) == len(frames) - int(censor), "guard count")
        pixels_last = None
        for j, e in enumerate(frames):
            rec = e["receipt"]
            path = Path(rec["path"])
            need(path.parent == directory and path.name == f"{episode}_{rec['phase']}_{rec['step']:02d}.png"
                 and path.is_file() and not path.is_symlink() and sha(path) == rec["file_sha256"],
                 "frame file binding")
            with Image.open(path) as im:
                pixels = np.asarray(im)
            need(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
                 and hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest() == rec["pixel_sha256"]
                 and rec["observed_event_index"] == e["index"] and rec["state"] == rec["snapshot"]["state"]
                 and rec["snapshot"]["bodies"]["agent"]["position"] == rec["state"][:2]
                 and rec["snapshot"]["bodies"]["block"]["position"] == rec["state"][2:4], "frame pixels/state")
            if rec["phase"] == "plan" and rec["step"] > 0:
                transition = returned[rec["step"]-1]
                need(transition["index"] < e["index"] and transition["state"] == rec["state"],
                     "frame transition")
            blue = int(np.count_nonzero(np.all(pixels == [65, 105, 225], axis=-1)))
            if not (censor and j == len(frames)-1):
                guard = guards[j]
                need(e["index"] < guard["index"] and guard["phase"] == rec["phase"]
                     and guard["step"] == rec["step"] and guard["canonical_agent_pixels"] == blue
                     and blue > 0 and all(0 <= v <= 512 for v in rec["state"][:4]), "guard proof")
            pixels_last = blue
        need(row["last_observation"] == frames[-1]["receipt"], "last frame")
        if n >= 5:
            rec = frames[16]["receipt"]
            need(row["prefix_verified"] is True and all(rec[k] == v for k, v in
                                                       plan["expected_prefix_checkpoint"].items()),
                 "prefix checkpoint")
        status = row["status"]
        if status == "completed15":
            need(n == 15 and not (returned[-1]["terminated"] or returned[-1]["truncated"])
                 and row["endpoint"] == frames[-1]["receipt"] and row["final_state"] == returned[-1]["state"],
                 "completed endpoint")
        elif status == "native_stopped":
            need(n > 0 and (returned[-1]["terminated"] or returned[-1]["truncated"])
                 and row["endpoint"] is None and row["final_state"] is None, "early stop")
        elif censor:
            r, rec, proof = returned[-1], frames[-1]["receipt"], row["censor"]
            outside = [{"coordinate": k, "value": r["state"][k]} for k in range(4)
                       if not 0 <= r["state"][k] <= 512]
            ce = [e for e in events if e["kind"] == "censored_out_of_domain"
                  and e["proof"]["branch"] == i]
            need(bool(outside) and len(ce) == 1 and ce[0]["proof"] == proof
                 and proof == {"branch": i, "step": n, "command": r["command"],
                               "returned_state": r["state"], "out_of_bounds": outside,
                               "file_sha256": rec["file_sha256"], "pixel_sha256": rec["pixel_sha256"],
                               "endpoint_accepted": False, "blue_pixels": pixels_last}
                 and frames[-1]["index"] < ce[0]["index"] < close["index"]
                 and row["endpoint"] is None and row["final_state"] is None, "censor evidence")
        else:
            raise ValueError("unsupported or failed branch status")
        endpoint = None if status != "completed15" else row["endpoint"]["state"]
        goal = frames[-1]["receipt"]["goal"]
        summaries.append({"branch": i, "arm": row["trajectory_arm"], "prefix": row["prefix_role"],
                          "suffix": row["suffix_role"], "status": status, "returned_steps": n,
                          "endpoint": endpoint, "goal": goal,
                          "contacts": sum(t["n_contacts"] for t in row["transitions"]),
                          "censor": row.get("censor")})
    need(sha(jp) == report["journal_sha256"] and sha(directory / "report.json") == REPORT_SHA
         and sha(prepared) == PREP_SHA, "input drift")
    return summaries


def analyze(rows):
    result = []
    for row in rows:
        item = dict(row)
        state, goal = row["endpoint"], row["goal"]
        item["object_position_error"] = None if state is None else math.hypot(state[2]-goal[2], state[3]-goal[3])
        item["object_angle_error"] = None if state is None else abs(math.atan2(
            math.sin(state[4]-goal[4]), math.cos(state[4]-goal[4])))
        result.append(item)
    index = {(r["arm"], r["prefix"], r["suffix"]): r for r in result}
    need(len(index) == 24, "analysis roster")
    comparisons = []
    def pair(kind, a, b):
        left, right = index[a], index[b]
        entry = {"kind": kind, "left_branch": left["branch"], "right_branch": right["branch"],
                 "left_status": left["status"], "right_status": right["status"]}
        for metric in ("object_position_error", "object_angle_error"):
            x, y = left[metric], right[metric]
            entry["left_minus_right_" + metric] = None if x is None or y is None else x-y
        comparisons.append(entry)
    for arm in ARMS:
        for suffix in SUFFIXES:
            for reference in ("zero", "greedy"):
                pair("same_suffix_selected_vs_" + reference,
                     (arm, "selected", suffix), (arm, reference, suffix))
        for prefix in PREFIXES:
            pair("saved_vs_zero_suffix", (arm, prefix, "saved"), (arm, prefix, "zero"))
    return {"fixed_branches": 24, "status_counts": dict(Counter(r["status"] for r in rows)),
            "rows": result, "comparisons": comparisons, "fixed_comparisons": 28,
            "saved_checks_passed": True, "P1_admission": False,
            "independent_confirmation": False, "fairness_proven": False,
            "source_closure_reverified": False,
            "limitations": "Known root, post-outcome exploratory hypothesis; H10 source closure not accessed. "
                           "Stopped/censored endpoints remain missing; no causal population inference."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    need(out.is_absolute() and out.name.startswith(NAME + "_") and not out.exists()
         and not out.is_symlink(), "exclusive output")
    result = analyze(audit(args.directory, args.prepared))
    result.update(report_sha256=REPORT_SHA, prepare_sha256=PREP_SHA, audit_source_sha256=sha(__file__))
    with out.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output": str(out), "sha256": sha(out), "counts": result["status_counts"]}))


if __name__ == "__main__":
    main()
