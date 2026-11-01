"""Execute the fixed real-midpoint multistart second packets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_uphill_midpoint_feedback_physical_20261031_main as prior
from . import p1_uphill_midpoint_multistart_plan_20261031 as planner

ROOT = prior.ROOT
NAME = "p1_uphill_midpoint_multistart_physical_20261031"
PLAN = ROOT / "results/p1_uphill_midpoint_multistart_plan_20261031_RUN01.json"
PLAN_SHA = "2d3c69d56e50a7a051e5316a56a1b90854446e3aecbb5e59ff44fdf6bed627b8"
need = prior.need


def verify_candidates(plan, entries):
    need(len(plan["rows"]) == len(entries) == 24, "candidate roster")
    for i, (row, (blind, _, raw, _, _)) in enumerate(zip(plan["rows"], entries, strict=True)):
        need(row["index"] == blind["index"] == i and len(row["candidate_hex"]) == len(row["candidate_scores"]) == 4,
             "candidate index")
        starts = planner.initial_actions(blind["logical_pair"], raw[1])
        need(hashlib.sha256(starts.tobytes()).hexdigest() == row["start_sha256"], "start identity")
        candidates = [np.frombuffer(bytes.fromhex(x), dtype="<f4") for x in row["candidate_hex"]]
        need(all(x.shape == (10,) and np.isfinite(x).all() and (np.abs(x) <= 1).all() for x in candidates)
             and np.isfinite(row["candidate_scores"]).all(), "candidate bounds")
        winner = min(range(4), key=lambda j: (row["candidate_scores"][j], j))
        need(row["selected_start"] == winner and row["optimized_score"] == row["candidate_scores"][winner],
             "winner score")
        selected = candidates[winner].tobytes() if row["optimized_score"] < row["baseline_score"] else raw[1].tobytes()
        need(bytes.fromhex(row["second_packet_hex"]) == selected, "winner bytes")


def configure():
    prior.PLAN, prior.PLAN_SHA, prior.NAME = PLAN, PLAN_SHA, NAME
    prior.planner.NAME = planner.NAME
    prior.configure()
    physical = prior.original.physical

    def build():
        result = prior.build()
        verify_candidates(json.loads(PLAN.read_bytes()), prior.planner.inputs()[0])
        for path in (Path(__file__).resolve(), Path(planner.__file__).resolve(), planner.PROTOCOL):
            result["bindings"][str(path)] = prior.sha(path)
        result["schema"] = NAME + ".prepare"
        return result

    physical.build = build
    return physical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "consume"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    parser.add_argument("--report-sha")
    parser.add_argument("--directory")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    physical = configure()
    out = Path(args.output)
    if args.command == "prepare":
        need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_prepare_"), "output")
        result = {"sha256": physical.design.run._write_new(out, physical.build())}
    elif args.command == "run":
        need(args.authorize_physics, "physics authorization")
        result = physical.execute(args.prepared, args.sha256, args.output)
    else:
        need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_consume_"), "output")
        result = {"sha256": physical.design.run._write_new(
            out, physical.consume(args.directory, args.report_sha, args.prepared, args.sha256))}
    print(json.dumps(result), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
