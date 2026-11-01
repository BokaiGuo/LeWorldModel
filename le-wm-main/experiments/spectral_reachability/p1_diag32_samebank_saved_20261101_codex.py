"""Read-only saved verification of the six diag32 physical branches."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_diag32_samebank_tail_20261101_codex as source
from . import p1_return_variants_analysis_20261101_main as evaluator

engine = source.engine


def verify(directory, expected):
    require = engine.require
    directory = Path(directory).resolve(strict=True)
    report_path = directory / "report.json"
    require(engine.sha(report_path) == expected, "external report SHA")
    report = json.loads(report_path.read_bytes())
    plan = json.loads((directory / "prepared.json").read_bytes())
    require(plan == source.build(), "frozen plan and complete source revalidation")
    require(report["status"] == "recorded_not_independently_consumed" and not report["errors"]
            and len(report["rows"]) == len(plan["schedule"]) == 6, "completed roster")
    journal = directory / "journal.jsonl"
    require(engine.sha(journal) == report["journal_sha256"], "journal SHA")
    events = engine.journal_reader.journal(journal.read_bytes(), report["journal_chain_tail"])
    counts = Counter()
    for event in events:
        if event["kind"] in ("operation_attempted", "operation_completed"):
            counts[(event["resource"], event["kind"].removeprefix("operation_"))] += event["units"]
    require(all(counts[(resource, kind)] == n for resource, values in report["counts"].items()
                for kind, n in values.items()), "operation accounting")
    rows, cp5 = [], {}
    for index, (row, slot) in enumerate(zip(report["rows"], plan["schedule"], strict=True)):
        require(row["episode_id"] == f"branch_{index:02}" and row["prefix_role"] == slot["prefix_role"]
                and row["suffix_role"] == slot["suffix_role"] and row["status"] in
                ("completed15", "native_stopped", "censored_out_of_domain"), "branch identity/status")
        require(json.loads((directory / f"branch_{index:02}_result.json").read_bytes()) == row, "row binding")
        n = row["control_returned"]
        require(row["executed_raw"] == slot["commands"][:n], "actions")
        issued = [e for e in events if e["kind"] == "branch_dispatch" and e["branch"] == index]
        returned = [e for e in events if e["kind"] == "branch_returned" and e["branch"] == index]
        require(len(issued) == len(returned) == n and all(
            a["step"] == b["step"] == j and a["command"] == b["command"] == slot["commands"][j-1]
            and a["index"] < b["index"] for j, (a, b) in enumerate(zip(issued, returned), 1)),
            "dispatch and return")
        obs = row["observations"]
        require(obs and [o["step"] for o in obs] == list(range(len(obs))) and
                all(obs[0][k] == plan["root"][k] for k in
                    ("state", "snapshot", "pixel_sha256", "goal")), "root/observation order")
        for point in obs:
            path = Path(point["path"]).resolve(strict=True)
            require(path.is_relative_to(directory) and engine.sha(path) == point["file_sha256"], "frame file")
            with Image.open(path) as image:
                pixel = np.asarray(image.convert("RGB"), np.uint8)
            require(hashlib.sha256(pixel.tobytes()).hexdigest() == point["pixel_sha256"]
                    and point["state"] == point["snapshot"]["state"]
                    and point["goal"] == plan["root"]["goal"], "state/goal/pixels")
            if point["step"]:
                require(point["state"] == returned[point["step"]-1]["state"], "return state")
        if len(obs) > 5:
            checkpoint = {k: obs[5][k] for k in ("state", "snapshot", "pixel_sha256")}
            role = slot["prefix_role"]
            if str(slot["candidate"]) in plan["prefixes"]:
                require(checkpoint == plan["prefixes"][str(slot["candidate"])], "historical cp5")
            if role in cp5:
                require(checkpoint == cp5[role], "matched prefix cp5")
            cp5[role] = checkpoint
        require((row["status"] != "completed15" or (n == 15 and len(obs) == 16))
                and (row["status"] != "native_stopped" or
                     (returned and (returned[-1]["terminated"] or returned[-1]["truncated"])))
                and (row["status"] != "censored_out_of_domain" or bool(row.get("censor"))),
                "stop evidence")
        last = obs[-1]
        native = evaluator.qualification.native(last["goal"], last["state"])
        rows.append(dict(prefix=slot["prefix_role"], tail=slot["suffix_role"], status=row["status"],
                         returned_steps=n, observed_step=last["step"],
                         joint_xy=native["joint_xy_error"], native_success=native["initial_success"],
                         metrics=evaluator.metrics(last["state"], last["goal"]), censor=row.get("censor"),
                         terminated=returned[-1]["terminated"] if returned else None,
                         truncated=returned[-1]["truncated"] if returned else None))
    return dict(status="saved_integrity_checked_not_independent_replay", report_sha256=expected,
                rows=rows, counts=report["counts"], independent_confirmation=False, P1_admission=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.directory, args.report_sha256)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
