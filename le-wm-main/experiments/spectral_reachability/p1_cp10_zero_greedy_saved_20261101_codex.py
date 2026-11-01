"""Read-only verification of fixed root0 zero/greedy physical branches."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_four_arm_prefix_intervention_20261030 as engine
from . import p1_return_variants_analysis_20261101_main as evaluator


def need(value, message):
    if not value:
        raise ValueError(message)


def verify(directory, report_sha):
    directory = Path(directory).resolve(strict=True)
    report_path = directory / "report.json"
    need(engine.sha(report_path) == report_sha, "report SHA")
    report = json.loads(report_path.read_bytes())
    plan = json.loads((directory / "prepared.json").read_bytes())
    need(plan["schema"] == "p1_cp10_zero_greedy_tail_20261101_codex.prepare.v1"
         and report["status"] == "recorded_not_independently_consumed" and not report["errors"]
         and len(report["rows"]) == len(plan["schedule"]) == 4, "study completeness")
    need(all(engine.sha(p) == h for p, h in plan["sources"].items()), "source closure")
    journal = directory / "journal.jsonl"
    need(engine.sha(journal) == report["journal_sha256"], "journal SHA")
    events = engine.journal_reader.journal(journal.read_bytes(), report["journal_chain_tail"])
    counts = Counter()
    for e in events:
        if e["kind"] in ("operation_attempted", "operation_completed"):
            counts[(e["resource"], e["kind"].removeprefix("operation_"))] += e["units"]
    need(all(counts[(resource, kind)] == n for resource, values in report["counts"].items()
             for kind, n in values.items()), "operation accounting")
    rows = []
    for i, (row, slot) in enumerate(zip(report["rows"], plan["schedule"], strict=True)):
        need(row["episode_id"] == f"branch_{i:02}" and row["prefix_role"] == slot["prefix_role"]
             and row["suffix_role"] == slot["suffix_role"]
             and row["status"] in ("completed15", "native_stopped", "censored_out_of_domain"),
             "branch identity and terminal status")
        need(json.loads((directory / f"branch_{i:02}_result.json").read_bytes()) == row,
             "row binding")
        n = row["control_returned"]
        need(row["executed_raw"] == slot["commands"][:n], "executed commands")
        issued = [e for e in events if e["kind"] == "branch_dispatch" and e["branch"] == i]
        returned = [e for e in events if e["kind"] == "branch_returned" and e["branch"] == i]
        need(len(issued) == len(returned) == n and all(
            a["step"] == b["step"] == j and a["command"] == b["command"] == slot["commands"][j-1]
            and a["index"] < b["index"] for j, (a, b) in enumerate(zip(issued, returned), 1)),
            "dispatch/return")
        obs = row["observations"]
        need(obs and all(obs[0][k] == plan["root"][k]
                         for k in ("state", "snapshot", "pixel_sha256", "goal")), "root identity")
        if n >= 5 and any(o["step"] == 5 for o in obs):
            cp = next(o for o in obs if o["step"] == 5)
            need(all(cp[k] == plan["prefixes"][slot["candidate"]][k]
                     for k in ("state", "snapshot", "pixel_sha256")), "cp5 identity")
        need([o["step"] for o in obs] == list(range(len(obs))), "observation order")
        for point in obs:
            path = Path(point["path"]).resolve(strict=True)
            need(path.is_relative_to(directory) and engine.sha(path) == point["file_sha256"], "image file")
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"), np.uint8)
            need(hashlib.sha256(pixels.tobytes()).hexdigest() == point["pixel_sha256"]
                 and point["state"] == point["snapshot"]["state"]
                 and point["goal"] == plan["root"]["goal"], "pixel/state/goal")
            if point["step"]:
                need(point["state"] == returned[point["step"]-1]["state"], "returned state")
        need((row["status"] != "completed15" or (n == 15 and len(obs) == 16))
             and (row["status"] != "censored_out_of_domain" or bool(row.get("censor")))
             and (row["status"] != "native_stopped" or
                  (returned and (returned[-1]["terminated"] or returned[-1]["truncated"]))),
             "stop evidence")
        last = obs[-1]
        native = evaluator.qualification.native(last["goal"], last["state"])
        rows.append(dict(prefix=slot["prefix_role"], tail=slot["suffix_role"], status=row["status"],
                         returned_steps=n, last_observed_step=last["step"],
                         metrics=evaluator.metrics(last["state"], last["goal"]),
                         joint_xy=native["joint_xy_error"], native_success=native["initial_success"],
                         censor=row.get("censor")))
    return dict(status="saved_integrity_checked_not_independent_replay", report_sha256=report_sha,
                rows=rows, counts=report["counts"], P1_admission=False,
                independent_confirmation=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", required=True)
    p.add_argument("--report-sha256", required=True)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    result = verify(args.directory, args.report_sha256)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
