"""Execute sealed trust-region plans with the existing fixed physical roster."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_uphill_midpoint_feedback_physical_20261031_main as prior
from . import p1_uphill_midpoint_trust_plan_20261031_main as planner

ROOT = prior.ROOT
NAME = "p1_uphill_midpoint_trust_physical_20261031_main"
PLAN = ROOT / "results/p1_uphill_midpoint_trust_plan_20261031_main_RUN01.json"
PLAN_SHA = "3a1cdfb981bce989c7ea77f5e9e9fae44d0afb6dc41a18f8e841491e91b25b48"
need = prior.need


def verify_radius(plan, entries):
    need(plan["radius"] == planner.RADIUS and len(plan["rows"]) == len(entries) == 24, "radius roster")
    for i, (row, (_, _, raw, _, _)) in enumerate(zip(plan["rows"], entries, strict=True)):
        second = np.frombuffer(bytes.fromhex(row["second_packet_hex"]), dtype="<f4")
        need(row["index"] == i and second.shape == (10,) and np.isfinite(second).all()
             and np.linalg.norm(second.astype(np.float64) - raw[1].reshape(-1)) <= planner.RADIUS + 1e-6,
             "trust radius")


def configure():
    prior.PLAN, prior.PLAN_SHA, prior.NAME = PLAN, PLAN_SHA, NAME
    prior.planner.NAME = planner.NAME
    prior.configure()
    physical = prior.original.physical

    def build():
        result = prior.build()
        verify_radius(json.loads(PLAN.read_bytes()), prior.planner.inputs()[0])
        for path in (Path(__file__).resolve(), planner.PROTOCOL):
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
        value = physical.consume(args.directory, args.report_sha, args.prepared, args.sha256)
        result = {"sha256": physical.design.run._write_new(out, value)}
    print(json.dumps(result), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
