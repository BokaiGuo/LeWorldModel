"""Saved physical crossover binding, censor handling, and paired metrics."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_four_prefix_saved_audit_20261030 as old
from . import e3_cp10_dls_crossover_20261031 as experiment
from . import e3_initial_goal_qualification_20261031 as qualification

NAME = "e3_cp10_dls_crossover_saved_20261031"
need, sha = old.need, old.sha
TAILS = ("zero", "structured", "native", "immediate", "random8_plus_zero2", "negative_structured")


def metrics(state, goal):
    state, goal = np.asarray(state, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    native = qualification.native(goal, state)
    return {"object_position": float(np.linalg.norm(state[2:4] - goal[2:4])),
            "pusher_position": float(np.linalg.norm(state[:2] - goal[:2])),
            "object_angle": float(abs(np.arctan2(np.sin(state[4] - goal[4]),
                                                 np.cos(state[4] - goal[4])))),
            "joint_xy": native["joint_xy_error"], "native_success": native["initial_success"]}


def check(prepared, prep_sha, directory, report_sha):
    prepared, directory = Path(prepared), Path(directory)
    need(sha(prepared) == prep_sha and sha(directory / "report.json") == report_sha, "input hashes")
    prep = json.loads(prepared.read_bytes())
    report = json.loads((directory / "report.json").read_bytes())
    need(report["status"] == "recorded_not_independently_consumed" and not report["errors"]
         and not report["pending"] and not report["secondary_errors"]
         and report["prepare_sha256"] == prep_sha and len(report["rows"]) == len(prep["schedule"]) == 24
         and not report["P1_admission"] and not report["fairness_proven"], "report")
    need(all(sha(path) == digest for path, digest in prep["bindings"].items()), "source drift")
    journal_path = directory / "journal.jsonl"
    need(sha(journal_path) == report["journal_sha256"], "journal SHA")
    events = old.journal(journal_path, report["journal_chain_tail"])
    need(events[0]["kind"] == "run_started" and events[-1]["kind"] == "run_finished"
         and events[-1]["status"] == report["status"], "journal boundaries")
    need(Counter(e["kind"] for e in events)["branch_closed"] == 24, "branch count")
    rows = []
    for index, (plan, row) in enumerate(zip(prep["schedule"], report["rows"], strict=True)):
        arm, tail = experiment.engine.ARMS[index // 6], TAILS[index % 6]
        need(plan["index"] == row["branch"] == index and row["trajectory_arm"] == arm
             and plan["trajectory_arm"] == arm and row["suffix_role"] == plan["suffix_role"] == tail
             and row["prefix_verified"] and row["initial_exact"] and not row["first_error"]
             and not row["secondary_errors"], "branch identity")
        expected = np.asarray(plan["commands"], dtype=np.float32)
        n = row["control_returned"]
        returned = np.asarray(row["executed_raw"], dtype=np.float32)
        need(expected.shape == (15, 2) and returned.shape == (n, 2)
             and np.array_equal(returned, expected[:n]) and len(row["transitions"]) == n,
             "executed action binding")
        transitions = row["transitions"]
        need(all(t["step"] == i and t["command"] == row["executed_raw"][i - 1]
                 for i, t in enumerate(transitions, 1))
             and all(not (t["terminated"] or t["truncated"]) for t in transitions[:-1]),
             "transition ordering")
        scope = [e for e in events if e.get("episode_id") == row["episode_id"]]
        frames = [e["receipt"] for e in scope if e["kind"] == "fresh_observation" and
                  e["receipt"]["phase"] == "plan"]
        need([f["step"] for f in frames] == list(range(n + 1)), "frame sequence")
        for frame in frames:
            path = Path(frame["path"])
            need(path.parent == directory and not path.is_symlink() and sha(path) == frame["file_sha256"],
                 "frame hash")
            with Image.open(path) as image:
                pixels = np.asarray(image)
            need(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
                 and old.hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()
                 == frame["pixel_sha256"], "frame pixels")
        need(all(frames[5][key] == value for key, value in plan["expected_prefix_checkpoint"].items()),
             "prefix replay")
        goal = frames[0]["goal"]
        need(all(frame["goal"] == goal for frame in frames), "goal identity")
        status = row["status"]
        need(status in ("completed15", "native_stopped", "censored_out_of_domain"), "branch status")
        accepted = status != "censored_out_of_domain"
        if status == "completed15":
            need(n == 15 and not (transitions[-1]["terminated"] or transitions[-1]["truncated"]),
                 "complete horizon")
        if status == "native_stopped":
            need(transitions[-1]["terminated"] or transitions[-1]["truncated"], "native stop")
        rows.append({"branch": index, "prefix": arm, "tail": tail, "status": status,
                     "returned_steps": n, "root": metrics(frames[0]["state"], goal),
                     "prefix_metrics": metrics(frames[5]["state"], goal),
                     "last_accepted": metrics(frames[-1]["state"], goal) if accepted else None,
                     "endpoint_h10": metrics(frames[15]["state"], goal) if accepted and n == 15 else None})
    for row in rows:
        zero = rows[experiment.engine.ARMS.index(row["prefix"]) * 6]["endpoint_h10"]
        endpoint = row["endpoint_h10"]
        row["delta_from_zero_h10"] = ({key: endpoint[key] - zero[key] for key in
                                       ("object_position", "pusher_position", "object_angle", "joint_xy")}
                                      if endpoint is not None and zero is not None else None)
    return {"schema": NAME, "status": "saved_checks_passed", "rows": rows,
            "prepared_sha256": prep_sha, "report_sha256": report_sha,
            "physical_event_chain_independently_replayed": False, "P1_admission": False,
            "independent_confirmation": False, "fairness_proven": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prepared", "prep-sha", "directory", "report-sha", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == experiment.ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists() and not output.is_symlink(),
         "exclusive output")
    result = check(args.prepared, args.prep_sha, args.directory, args.report_sha)
    digest = experiment.engine.raw(result)
    with output.open("xb") as stream:
        need(stream.write(digest) == len(digest), "output write")
    print(json.dumps({"status": result["status"], "rows": len(result["rows"]), "sha256": sha(output)}))


if __name__ == "__main__":
    main()
