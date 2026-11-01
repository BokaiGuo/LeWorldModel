"""Validate all physical branches and report uncensored candidate coverage."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_candidate_intervention_20261101_main as run
from . import e3_initial_goal_qualification_20261031 as qualification
from .p1_common_h10_analysis_20261101_main import metrics, delta


def physical(state, goal):
    result = qualification.native(goal, state)
    return dict(**metrics(state, goal), joint_xy=result["joint_xy_error"],
                native7=result["native7_distance"])


def analyze(directory):
    directory = Path(directory)
    engine = run.engine
    report = json.loads((directory / "report.json").read_bytes())
    plan = json.loads((directory / "prepared.json").read_bytes())
    engine.require(report["status"] == "recorded_not_independently_consumed"
                   and len(report["rows"]) == len(plan["schedule"]) == 24, "complete roster")
    engine.require(engine.sha(directory / "journal.jsonl") == report["journal_sha256"], "journal bytes")
    journal = engine.journal_reader.journal((directory / "journal.jsonl").read_bytes(),
                                             report["journal_chain_tail"])
    engine.require(all(engine.sha(p) == h for p, h in plan["bindings"].items()), "bound source bytes")
    rows, prefixes, images = [], {}, 0
    for row, branch in zip(report["rows"], plan["schedule"], strict=True):
        engine.require(row["branch"] == branch["index"] and row["candidate"] == branch["candidate"]
                       and row["suffix_role"] == branch["suffix_role"], "branch identity")
        count = row["control_returned"]
        engine.require(row["executed_raw"] == branch["commands"][:count], "actual actions")
        scope = [e for e in journal if e.get("branch") == branch["index"]]
        sent = [e for e in scope if e["kind"] == "branch_dispatch"]
        returned = [e for e in scope if e["kind"] == "branch_returned"]
        engine.require(len(sent) == len(returned) == count == len(row["transitions"]), "dispatch/return count")
        for index, (a, b, t) in enumerate(zip(sent, returned, row["transitions"], strict=True), 1):
            engine.require(a["step"] == b["step"] == t["step"] == index
                           and a["command"] == b["command"] == t["command"] == branch["commands"][index - 1]
                           and all(b[k] == t[k] for k in t), "physical event identity")
        observations = list(row["observations"])
        if row["status"] == "censored_out_of_domain":
            observations.append(row["last_observation"])
            engine.require(engine.validate_censor(branch, row["transitions"][-1], observations[-1])
                           == row["censor"], "domain censorship")
        engine.require([x["step"] for x in observations] == list(range(count + 1)),
                       "all returned physical frames")
        goal = observations[0]["goal"]
        for cp in observations:
            path = Path(cp["path"])
            engine.require(path.parent == directory and engine.sha(path) == cp["file_sha256"], "image file")
            with Image.open(path) as image:
                pixels = np.asarray(image)
            engine.require(engine.old.array_sha(pixels) == cp["pixel_sha256"], "image pixels")
            engine.require(cp["goal"] == goal, "fixed goal")
            if cp["step"]:
                engine.require(cp["state"] == row["transitions"][cp["step"] - 1]["state"], "frame/state alignment")
            images += 1
        cp5 = next((x for x in observations if x["step"] == 5), None)
        if cp5 and row.get("prefix_checkpoint"):
            checkpoint = {k: cp5[k] for k in ("state", "snapshot", "pixel_sha256")}
            engine.require(checkpoint == row["prefix_checkpoint"], "cp5 pointer")
            if row["candidate"] in prefixes:
                engine.require(prefixes[row["candidate"]] == checkpoint, "repeated prefix equivalence")
            prefixes[row["candidate"]] = checkpoint
        initial = physical(observations[0]["state"], goal)
        endpoint = physical(row["endpoint"]["state"], goal) if row["endpoint"] else None
        success_steps = [t["step"] for t in row["transitions"]
                         if qualification.native(goal, t["state"])["initial_success"]]
        rows.append(dict(candidate=row["candidate"], suffix=row["suffix_role"], status=row["status"],
                         steps=count, native_success_steps=success_steps,
                         initial=initial,
                         prefix_delta=delta(physical(cp5["state"], goal), initial) if cp5 else None,
                         endpoint15=endpoint, last_observed=physical(observations[-1]["state"], goal)))
    for row in rows:
        zero = next(r for r in rows if r["candidate"] == row["candidate"] and r["suffix"] == "zero")
        row["endpoint_minus_own_zero"] = (delta(row["endpoint15"], zero["endpoint15"])
                                           if row["endpoint15"] and zero["endpoint15"] else None)
    return dict(rows=rows, checked_images=images, predicted_scores=plan["predicted_scores"],
                model_selected=plan["selected_index"], P1_admission=False, independent_confirmation=False,
                sources={str(directory / name): engine.sha(directory / name)
                         for name in ("report.json", "prepared.json", "journal.jsonl")},
                interpretation="Full developmental candidate diagnosis; no outcome-selected deployment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(rows=len(result["rows"]), images=result["checked_images"], output=str(args.output))))
