"""Verify successful and failed branches of an exclusive all-candidate replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from . import p3_all30_cpu_physical_20261102_codexexclusive as replay

require = replay.require
moving = replay.moving
NAME = "p3_all30_cpu_consume_20261102_codexexclusive"


def chain(path, expected_tail):
    require(path.stat().st_size < replay.CAP, "journal byte cap")
    events, previous = [], None
    with path.open("rb") as f:
        for i, line in enumerate(f):
            require(line.endswith(b"\n"), "incomplete journal line")
            e = json.loads(line)
            require(e["index"] == i and e["previous_sha256"] == previous
                    and e["event_sha256"] == moving.source.digest(
                        {k: v for k, v in e.items() if k != "event_sha256"}), "journal chain")
            previous = e["event_sha256"]
            events.append(e)
    require(events and previous == expected_tail and events[-1]["kind"] == "run_finished",
            "terminal journal tail")
    return events


def validate(p, data, report, events, directory):
    require(events[0]["kind"] == "run_started" and events[0]["prepare"] == p, "prepare event")
    require(len(p["plans"]) == len(report["rows"]) == 30 and report["contract"] == p["contract"],
            "report roster/contract")
    completed = [e for e in events if e["kind"] == "operation_completed"]
    attempted = [e for e in events if e["kind"] == "operation_attempted"]
    acount = Counter(e["resource"] for e in attempted)
    ccount = Counter(e["resource"] for e in completed)
    require(set(acount) <= set(p["contract"]["caps"]) and set(ccount) <= set(p["contract"]["caps"]),
            "unknown resource")
    for resource, cap in p["contract"]["caps"].items():
        require(report["counts"][resource] == {"attempted": acount[resource], "completed": ccount[resource]}
                and 0 <= ccount[resource] <= acount[resource] <= cap, "resource census")
    require(events[-1]["counts"] == report["counts"] and events[-1]["errors"] == report["errors"]
            and events[-1]["status"] == report["status"], "terminal report")
    closed = [e for e in events if e["kind"] == "branch_closed"]
    started_count = 0
    failed = False
    png_count = 0
    for i, (plan, row) in enumerate(zip(p["plans"], report["rows"], strict=True)):
        branch = [e for e in events if e.get("branch") == i]
        if row["status"] == "not_started":
            require(not branch, "unstarted branch has events")
            continue
        require(not failed and i == started_count, "noncontiguous execution")
        started_count += 1
        start = [e for e in branch if e["kind"] == "branch_started"]
        close = [e for e in branch if e["kind"] == "branch_closed"]
        require(len(start) == len(close) == 1 and start[0]["plan"] == plan
                and close[0]["receipt"] == row and
                start[0]["index"] < close[0]["index"] and
                all(e["index"] <= close[0]["index"] for e in branch), "branch binding")
        require((row["root_slot"], row["candidate_index"], row["selection_digest"]) ==
                (plan["root_slot"], plan["candidate_index"], plan["selection_digest"]), "row identity")
        observations = [e["receipt"] for e in branch if e["kind"] == "fresh_observation"]
        guards = [e for e in branch if e["kind"] == "guard_receipt"]
        require(len(observations) == row["PNG"] == len(guards), "observation and guard count")
        for obs, guard in zip(observations, guards, strict=True):
            require((obs["phase"], obs["step"]) == (guard["phase"], guard["step"]), "guard order")
            path = directory / obs["png"]
            require(path.parent == directory and moving.sha(path) == obs["file_sha256"], "PNG SHA")
            png_count += 1
        warm = [e for e in branch if e["kind"] == "warmup_return"]
        require([e["step"] for e in warm] == list(range(1, len(warm) + 1)) and
                [e["raw_action"] for e in warm] == start[0]["raw_warmup"][:len(warm)], "warmup actions")
        trans = [e for e in branch if e["kind"] == "returned_transition" and e["phase"] == "plan"]
        require([e["step"] for e in trans] == list(range(1, row["own_returned"] + 1)) and
                [e["command"] for e in trans] == plan["commands"][:len(trans)], "plan actions")
        if row["initial_exact"]:
            ref = data["references"][plan["root_slot"]]
            checkpoints = [o for o in observations if o["phase"] == "warmup" and o["step"] in (0, 5, 10)]
            require(len(checkpoints) == 3 and len(warm) == 10, "root history checkpoints")
            for j, obs in enumerate(checkpoints):
                require(obs["step"] == (0, 5, 10)[j] and
                        obs["pixel_sha256"] == moving.moving.image_sha(ref["history"][j]) and
                        obs["state"] == ref["states"][j].tolist() and obs["snapshot"] == ref["dynamics"][j],
                        "root RGB/state/snapshot")
            require(row["initial_metrics"] == moving.metric(row["goal"], row["root_state"]), "initial metric")
            attached = [e for e in branch if e["kind"] == "goal_attached"]
            require(len(attached) == 1 and attached[0]["goal"] == row["goal"] == ref["goal"].tolist()
                    and (not trans or attached[0]["index"] < trans[0]["index"]), "goal before plan")
        require(not trans or row["initial_exact"], "no plan before root authentication")
        if trans:
            require(trans[-1]["state"] == row["final_state"] and
                    row["final_metrics"] == moving.metric(row["goal"], row["final_state"]), "endpoint")
        for checkpoint in row["checkpoints"]:
            require(checkpoint in observations, "checkpoint receipt")
        if row["status"] == "failed":
            failed = True
            require(row["first_error"] is not None, "failed branch reason")
        else:
            require(row["status"] in ("horizon_completed", "native_success", "native_stopped")
                    and row["initial_exact"] and row["first_error"] is None
                    and all(g["accepted"] and g["render_preserved_dynamics"] for g in guards), "complete branch")
    require(len(closed) == started_count and [e["branch"] for e in closed] == list(range(started_count)),
            "closed prefix")
    if failed:
        require(report["status"] == "partial_or_failed" and report["errors"], "failure must be retained")
    else:
        require(started_count == 30 and report["status"] == "all_30_recorded" and not report["errors"],
                "all branches complete")
    return {"status": "partial_record_verified" if failed else "all_30_record_verified",
            "completed_branches": sum(r["status"] in ("horizon_completed", "native_success", "native_stopped")
                                      for r in report["rows"]),
            "failed_branches": sum(r["status"] == "failed" for r in report["rows"]),
            "not_started": sum(r["status"] == "not_started" for r in report["rows"]),
            "png_verified": png_count, "independent_physical_attestation": False}


def consume(prepared, prepare_sha, directory, report_sha):
    p, data = replay.verify(prepared, prepare_sha)
    directory = Path(directory).resolve()
    require(moving.read(directory / "execution_binding.json") ==
            {"prepared": str(Path(prepared).resolve()), "prepare_sha256": prepare_sha}, "execution binding")
    require(moving.sha(directory / "report.json") == report_sha, "report SHA")
    report = moving.read(directory / "report.json")
    require(moving.sha(directory / "journal.jsonl") == report["journal_sha256"], "journal SHA")
    events = chain(directory / "journal.jsonl", report["journal_chain_tail"])
    result = validate(p, data, report, events, directory)
    result.update(prepare_sha256=prepare_sha, report_sha256=report_sha,
                  journal_sha256=report["journal_sha256"], errors=report["errors"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepare-sha", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--report-sha", required=True)
    parser.add_argument("--output", required=True)
    a = parser.parse_args()
    result = consume(a.prepared, a.prepare_sha, a.run, a.report_sha)
    out = replay.destination(a.output, "_CONSUME_", ".json")
    with out.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"result": result, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
