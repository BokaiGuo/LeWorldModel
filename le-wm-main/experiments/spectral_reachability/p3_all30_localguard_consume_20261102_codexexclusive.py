"""CPU-only verification of fixed-denominator local-guard replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from . import p3_all30_localguard_cpu_20261102_codexexclusive as run
from . import p3_all30_cpu_consume_20261102_codexexclusive as prior

moving = run.moving
require = run.require


def validate(p, refs, report, events, directory):
    require(events[0]["kind"] == "run_started" and events[0]["prepare"] == p, "prepare event")
    require(len(p["plans"]) == len(report["rows"]) == report["fixed_denominator"] == 30 and
            report["contract"] == p["contract"], "roster and contract")
    require(events[-1]["kind"] == "run_finished" and events[-1]["status"] == report["status"] and
            events[-1]["errors"] == report["errors"] and events[-1]["counts"] == report["counts"] and
            events[-1]["candidate_counts"] == report["candidate_counts"], "terminal report")
    attempts = Counter(e["resource"] for e in events if e["kind"] == "operation_attempted")
    completed = Counter(e["resource"] for e in events if e["kind"] == "operation_completed")
    require(set(attempts) <= p["contract"]["caps"].keys() and
            set(completed) <= p["contract"]["caps"].keys(), "resource names")
    for key, cap in p["contract"]["caps"].items():
        require(report["counts"][key] == {"attempted": attempts[key], "completed": completed[key]}
                and completed[key] <= attempts[key] <= cap, "resource counts/caps")
    counts = Counter(r["status"] for r in report["rows"])
    require(all(report["candidate_counts"][key] == counts[key] for key in report["candidate_counts"]) and
            sum(counts.values()) == 30 and set(counts) <= set(report["candidate_counts"]), "candidate counts")
    valid = {"horizon_completed", "native_success", "native_stopped"}
    require(report["valid_endpoint_count"] == sum(counts[s] for s in valid) and
            report["native_successes_valid"] == sum(r["native_success"] for r in report["rows"]
                                                    if r["status"] in valid), "valid denominator")
    closed = [e for e in events if e["kind"] == "branch_closed"]
    require([e["branch"] for e in closed] == list(range(len(closed))), "ordered closure")
    png_count = 0
    for i, (plan, row) in enumerate(zip(p["plans"], report["rows"], strict=True)):
        branch = [e for e in events if e.get("branch") == i]
        if row["status"] == "not_started":
            require(not branch and i >= len(closed), "unstarted branch")
            continue
        require(i < len(closed) and closed[i]["receipt"] == row and
                (row["root_slot"], row["candidate_index"], row["selection_digest"]) ==
                (plan["root_slot"], plan["candidate_index"], plan["selection_digest"]), "row closure")
        starts = [e for e in branch if e["kind"] == "branch_started"]
        if not starts:
            require(row["status"] == "failed" and i == len(closed)-1, "setup failure")
            continue
        require(len(starts) == 1 and starts[0]["plan"] == plan and
                all(e["index"] <= closed[i]["index"] for e in branch), "branch order")
        obs = [e["receipt"] for e in branch if e["kind"] == "fresh_observation"]
        guards = [e for e in branch if e["kind"] == "guard_receipt"]
        require(len(obs) == len(guards) == row["PNG"], "observations and guards")
        for o, g in zip(obs, guards, strict=True):
            path = directory / o["png"]
            require(path.parent == directory and not path.is_symlink() and
                    moving.sha(path) == o["file_sha256"] and
                    (o["phase"], o["step"]) == (g["phase"], g["step"]), "frame receipt")
            png_count += 1
        warm = [e for e in branch if e["kind"] == "warmup_return"]
        require([e["step"] for e in warm] == list(range(1, len(warm)+1)) and
                [e["raw_action"] for e in warm] == starts[0]["raw_warmup"][:len(warm)],
                "warmup commands")
        transitions = [e for e in branch if e["kind"] == "returned_transition" and e["phase"] == "plan"]
        require([e["step"] for e in transitions] == list(range(1, row["own_returned"]+1)) and
                [e["command"] for e in transitions] == plan["commands"][:len(transitions)] and
                row["own_attempted"] >= row["own_returned"], "plan commands")
        if row["initial_exact"]:
            ref = refs[plan["root_slot"]]
            snapshots = [o for o in obs if o["phase"] == "warmup" and o["step"] in (0, 5, 10)]
            require(len(snapshots) == 3 and len(warm) == 10, "root boundaries")
            for j, o in enumerate(snapshots):
                require(o["step"] == (0, 5, 10)[j] and o["pixel_sha256"] ==
                        moving.moving.image_sha(ref["history"][j]) and
                        o["state"] == ref["states"][j].tolist() and o["snapshot"] == ref["dynamics"][j],
                        "root RGB/state/snapshot")
            attached = [e for e in branch if e["kind"] == "goal_attached"]
            require(len(attached) == 1 and attached[0]["goal"] == row["goal"] == ref["goal"].tolist()
                    and (not transitions or attached[0]["index"] < transitions[0]["index"])
                    and row["initial_metrics"] == moving.metric(row["goal"], row["root_state"]), "goal/root")
        require(not transitions or row["initial_exact"], "plan before root guard")
        if transitions:
            require(transitions[-1]["state"] == row["final_state"] and
                    row["final_metrics"] == moving.metric(row["goal"], row["final_state"]), "endpoint")
        require(all(c in obs for c in row["checkpoints"]), "checkpoint membership")
        rejected = [g for g in guards if not g["accepted"]]
        if row["status"] == "guard_failed_invalid":
            require(len(rejected) == 1 and rejected[-1] == guards[-1] and
                    run.local_guard(dict(row, status="failed"), rejected[0], i), "local guard classification")
            require(not any(e["kind"] == "operation_attempted" and e["resource"] == "public_step" and
                            e["index"] > rejected[0]["index"] for e in branch), "no dispatch after guard")
        elif row["status"] in valid:
            require(not rejected and all(g["render_preserved_dynamics"] for g in guards)
                    and row["first_error"] is None and row["initial_exact"], "valid candidate guards")
        else:
            require(row["status"] == "failed" and i == len(closed)-1 and report["errors"], "global failure")
    if report["status"] == "all_candidates_processed":
        require(len(closed) == 30 and not report["errors"] and counts["not_started"] == counts["failed"] == 0,
                "all processed")
    else:
        require(report["status"] == "partial_or_failed" and report["errors"], "partial status")
    return {"status": "all_records_verified" if report["status"] == "all_candidates_processed"
            else "partial_records_verified", "fixed_denominator": 30,
            "candidate_counts": report["candidate_counts"], "valid_endpoint_count": report["valid_endpoint_count"],
            "native_successes_valid": report["native_successes_valid"], "png_verified": png_count,
            "independent_physical_attestation": False}


def consume(prepared, prepare_sha, directory, report_sha):
    p, data = run.verify(prepared, prepare_sha)
    directory = Path(directory).resolve()
    require(moving.read(directory / "execution_binding.json") ==
            {"prepared": str(Path(prepared).resolve()), "prepare_sha256": prepare_sha}, "execution binding")
    require(moving.sha(directory / "report.json") == report_sha, "report SHA")
    report = moving.read(directory / "report.json")
    journal = directory / "journal.jsonl"
    require(moving.sha(journal) == report["journal_sha256"], "journal SHA")
    events = prior.chain(journal, report["journal_chain_tail"])
    result = validate(p, data["references"], report, events, directory)
    result.update(prepare_sha256=prepare_sha, report_sha256=report_sha, journal_sha256=report["journal_sha256"])
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
    out = run.destination(a.output, "_CONSUME_", ".json")
    with out.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"result": result, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
