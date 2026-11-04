"""Fixed RGB outcome metrics on the sealed six-chain Reacher diagnostic."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from . import reacher_cem32_same_chain_rgb_20261102 as rgb

host, setup, ROOT = rgb.host, rgb.setup, rgb.ROOT
RGB_REPORT = ROOT / "results/reacher_cem32_same_chain_rgb_SAVED_20261102_v1.json"
RGB_SHA = "a22e497bae27ad74efabe2af437fb84811b1d2f683efa31420c731df3604e76b"
PROTOCOL = ROOT.parent / "docs/PC_REACHER_ACTUAL_RGB_PIXEL_METRIC_PROTOCOL_20261102.md"


def metrics(frame, goal):
    a, b = np.asarray(frame), np.asarray(goal)
    host.require(a.dtype == b.dtype == np.uint8 and a.shape == b.shape and a.ndim == 3
                 and a.shape[-1] == 3 and a.size > 0, "invalid RGB")
    difference = (a.astype(np.float64) - b.astype(np.float64)) / 255.0
    return {"pixel_mse": float(np.mean(difference * difference)),
            "pixel_mae": float(np.mean(np.abs(difference)))}


def summarize(rows, metric):
    physical = "physical_final"
    pairs = [tuple(np.sign(row1[metric] - row2[metric])
                   for metric in (metric, physical))
             for row1, row2 in itertools.combinations(rows, 2)]
    chosen = min(rows, key=lambda row: (row[metric], row["order"]))
    best = min(row[physical] for row in rows)
    return {"chosen_role": chosen["role"], "physical_regret": chosen[physical] - best,
            "concordant": int(sum(a == b and a != 0 for a, b in pairs)),
            "discordant": int(sum(a * b < 0 for a, b in pairs)),
            "ties": int(sum(a == 0 or b == 0 for a, b in pairs)), "pairs": len(pairs)}


def run():
    host.require(setup._sha256_file(RGB_REPORT) == RGB_SHA, "RGB receipt hash")
    rgb.prepare()
    physical_plan = json.loads(rgb.PLAN.read_text())
    physical_events = rgb.exp.run.saved.verify_journal(rgb.JOURNAL)
    actual = json.loads(RGB_REPORT.read_text())
    host.require(actual["status"] == "saved_consistency_complete" and len(actual["rows"]) == 6,
                 "actual encoded receipt")
    old = json.loads(rgb.exp.run.physical.OLD_PLAN.read_text())
    source = rgb.sensitivity.contexts(
        rgb.exp.run.saved.verify_journal(rgb.sensitivity.JOURNAL), old["roots"])
    ctx = rgb.exp.run.saved.ctx
    output = []
    pos = 0
    for root in physical_plan["roots"]:
        goal = rgb.bound_goal(source, root, ctx)
        rows = []
        for i, branch in enumerate(root["branches"]):
            found = [e for e in physical_events if e["kind"] == "branch_returned"
                     and e["seed"] == root["seed"] and e["branch_index"] == i]
            host.require(len(found) == 1, "physical branch")
            points = found[0]["outcome"]["trajectory"]
            frame = ctx.unpack_rgb_observation(points[15]["rgb"])
            saved = actual["rows"][pos]
            pos += 1
            host.require(saved["seed"] == root["seed"] and saved["role"] == branch["role"]
                         and saved["physical_final"] == points[15]["qpos_distance"], "paired binding")
            rows.append({"order": i, **saved, **metrics(frame, goal),
                         "final_rgb_sha256": ctx.rgb_digest(frame), "goal_rgb_sha256": ctx.rgb_digest(goal)})
        output.append({"seed": root["seed"], "rows": rows,
                       "comparisons": {name: summarize(rows, name) for name in
                                       ("predicted_cost", "actual_cost", "pixel_mse", "pixel_mae")}})
    host.require(pos == 6, "arm denominator")
    files = [Path(__file__), PROTOCOL, RGB_REPORT, rgb.PLAN, rgb.JOURNAL, rgb.RECEIPT,
             rgb.sensitivity.JOURNAL]
    return {"schema": "reacher_actual_rgb_pixel_metric_v1",
            "files": {str(p): setup._sha256_file(p) for p in files}, "roots": output,
            "independent_confirmation": False, "P1_admission": False,
            "scope": "retrospective real-RGB evaluation; not a deployable selector"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run()
    host.write_new(args.output, result)
    print(json.dumps({"sha256": setup._sha256_file(Path(args.output)), "roots": result["roots"]}))


if __name__ == "__main__":
    main()
