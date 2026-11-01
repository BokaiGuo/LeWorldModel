"""Independent read-only saved verification for cp10 fixed-tail exploration."""
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
    need(engine.sha(report_path) == report_sha, "external report SHA")
    report = json.loads(report_path.read_bytes())
    plan = json.loads((directory / "prepared.json").read_bytes())
    need(report["status"] == "recorded_not_independently_consumed" and not report["errors"]
         and report["P1_admission"] is False and report["independent_confirmation"] is False
         and report["neural_calls"] == 0 and len(report["rows"]) == len(plan["schedule"]) == 4,
         "report completeness")
    need(all(engine.sha(p) == digest for p, digest in plan["sources"].items()), "source closure")
    journal = directory / "journal.jsonl"
    need(engine.sha(journal) == report["journal_sha256"], "journal SHA")
    events = engine.journal_reader.journal(journal.read_bytes(), report["journal_chain_tail"])
    counts = Counter()
    for e in events:
        if e["kind"] in ("operation_attempted", "operation_completed"):
            counts[(e["resource"], "attempted" if e["kind"] == "operation_attempted" else "completed")] += e["units"]
    need(all(counts[(resource, kind)] == n for resource, value in report["counts"].items()
             for kind, n in value.items()), "physical counts")
    rows, images = [], 0
    for index, (row, slot) in enumerate(zip(report["rows"], plan["schedule"], strict=True)):
        need(row["episode_id"] == f"branch_{index:02}" and row["status"] == "completed15"
             and row["control_returned"] == 15 and row["prefix_role"] == slot["prefix_role"]
             and row["suffix_role"] == slot["suffix_role"] and row["executed_raw"] == slot["commands"]
             and len(row["observations"]) == 16, "branch action/roster")
        need(json.loads((directory / f"branch_{index:02}_result.json").read_bytes()) == row, "branch report binding")
        issued = [e for e in events if e["kind"] == "branch_dispatch" and e["branch"] == index]
        returned = [e for e in events if e["kind"] == "branch_returned" and e["branch"] == index]
        need(len(issued) == len(returned) == 15 and all(
            a["step"] == b["step"] == i and a["command"] == b["command"] == slot["commands"][i-1]
            and a["index"] < b["index"] and not b["terminated"] and not b["truncated"]
            for i, (a, b) in enumerate(zip(issued, returned, strict=True), 1)), "dispatch/return journal")
        obs = row["observations"]
        need([x["step"] for x in obs] == list(range(16)), "observation sequence")
        need(all(obs[0][k] == plan["root"][k] for k in ("state", "snapshot", "pixel_sha256", "goal")),
             "physical root")
        need(all(obs[5][k] == plan["prefixes"][slot["candidate"]][k]
                 for k in ("state", "snapshot", "pixel_sha256")), "cp5 replay identity")
        for i, point in enumerate(obs):
            path = Path(point["path"]).resolve(strict=True)
            need(path.is_relative_to(directory) and engine.sha(path) == point["file_sha256"],
                 "frame path/file SHA")
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"), np.uint8)
            need(hashlib.sha256(pixels.tobytes()).hexdigest() == point["pixel_sha256"], "frame pixel SHA")
            need(point["state"] == point["snapshot"]["state"] and point["goal"] == plan["root"]["goal"],
                 "observation state/goal")
            if i:
                need(point["state"] == returned[i-1]["state"], "returned observation state")
            images += 1
        endpoint = obs[-1]
        native = evaluator.qualification.native(endpoint["goal"], endpoint["state"])
        rows.append(dict(prefix=row["prefix_role"], tail=row["suffix_role"],
                         step=15, metrics=evaluator.metrics(endpoint["state"], endpoint["goal"]),
                         joint_xy=native["joint_xy_error"], native_success=native["initial_success"]))
    values = {(r["prefix"], r["tail"]): r["joint_xy"] for r in rows}
    effect_s = values["structured", "zero"] - values["structured", "common"]
    effect_c = values["cem", "zero"] - values["cem", "common"]
    return dict(status="saved_integrity_checked_not_scientific_admission", report_sha256=report_sha,
                protocol_sha256=engine.sha(directory / "prepared.json"), images_verified=images,
                physics_counts=report["counts"], rows=rows,
                effects=dict(structured_zero_minus_common=effect_s, cem_zero_minus_common=effect_c,
                             structured_minus_cem_interaction=effect_s-effect_c),
                independent_confirmation=False, P1_admission=False,
                limitations="one known post-outcome root; no independent simulator replay or kernel accounting")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.directory, args.report_sha256)
    if args.output:
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
