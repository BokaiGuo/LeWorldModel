"""Physically replay fixed packet plans with a fail-closed midpoint guard."""
import argparse
import hashlib
import json
from pathlib import Path

from . import p1_packet_midpoint_feedback_plan_20261031 as planner
from . import p1_uphill_midpoint_feedback_physical_20261031_main as prior

ROOT = planner.source.ROOT
NAME = "p1_packet_midpoint_feedback_physical_20261031"
PLAN = ROOT / "results/p1_packet_midpoint_feedback_plan_20261031_RUN01.json"
PLAN_SHA = "af124ef8911443e8874ac1ec9696328c5497316277be6a6feccaf1a07e73c2c1"
OLD = planner.source.PREP
OLD_SHA = planner.source.PREP_SHA
physical = prior.original.physical
need, sha = physical.need, physical.sha


def build():
    need(sha(PLAN) == PLAN_SHA and sha(OLD) == OLD_SHA, "input hashes")
    old, plan = json.loads(OLD.read_bytes()), json.loads(PLAN.read_bytes())
    inputs, sources = planner.inputs()
    need(plan["schema"] == planner.NAME and plan["status"] == "complete"
         and plan["model_before"] == plan["model_after"], "plan identity")
    compatible = dict(plan, schema=prior.planner.NAME)
    roster = prior.assemble(old, compatible, inputs)
    for item, (blind, pair, raw, pixels, frame) in zip(roster, inputs, strict=True):
        need(frame["pixel_sha256"] == hashlib.sha256(pixels.tobytes()).hexdigest(),
             "midpoint pixel binding")
        item["midpoint"] = {k: frame[k] for k in ("pixel_sha256", "state", "snapshot")}
    bindings = dict(old["bindings"])
    for mapping in (plan["sources"], sources):
        for path, digest in mapping.items():
            need(path not in bindings or bindings[path] == digest, "binding collision")
            bindings[path] = digest
    for path in (PLAN, OLD, Path(__file__).resolve()):
        bindings[str(path)] = sha(path)
    need(all(sha(p) == digest for p, digest in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "roster": roster, "bindings": bindings,
            "plan_sha256": PLAN_SHA, "fixed_rows": 24, "P1_admission": False,
            "independent_confirmation": False, "fairness_proven": False}


def guarded(original_execute, anchor, item, data, path, factory):
    base = physical.engine.h10.base
    original_capture = base.capture
    seen = []

    def capture(env, obs, phase, step, row, meter, store, reference=None):
        pixels = original_capture(env, obs, phase, step, row, meter, store, reference)
        if phase == "probe" and step == 5:
            receipt = row["trajectory"][-1]
            need(all(receipt[k] == item["midpoint"][k] for k in ("pixel_sha256", "state", "snapshot"))
                 and hashlib.sha256(pixels.tobytes()).hexdigest() == item["midpoint"]["pixel_sha256"],
                 "midpoint mismatch before dispatch")
            seen.append(True)
        return pixels

    base.capture = capture
    try:
        report = original_execute(anchor, item, data, path, factory)
    finally:
        base.capture = original_capture
    if len(report["row"]["responses"]) >= 5:
        need(seen == [True], "midpoint guard absent")
    return report


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
    physical.NAME, physical.engine.NAME, physical.build = NAME, NAME, build
    out = Path(args.output)
    if args.command == "prepare":
        need(out.is_absolute() and out.parent == ROOT / "results"
             and out.name.startswith(NAME + "_prepare_"), "output")
        result = {"sha256": physical.design.run._write_new(out, build())}
    elif args.command == "run":
        need(args.authorize_physics, "physical authorization")
        original_execute = physical.engine.execute
        physical.engine.execute = lambda *a: guarded(original_execute, *a)
        try:
            result = physical.execute(args.prepared, args.sha256, args.output)
        finally:
            physical.engine.execute = original_execute
    else:
        need(out.is_absolute() and out.parent == ROOT / "results"
             and out.name.startswith(NAME + "_consume_"), "output")
        value = physical.consume(args.directory, args.report_sha, args.prepared, args.sha256)
        fixed = build()
        for item, row in zip(fixed["roster"], value["rows"], strict=True):
            if row["returned_suffix"] >= 5:
                report = json.loads((Path(args.directory) / f"row_{item['index']:02d}" / "report.json").read_bytes())
                frame = planner.source.earlier.frame_at(report["row"], 5, item["condition"])
                need(all(frame[k] == item["midpoint"][k] for k in ("pixel_sha256", "state", "snapshot")),
                     "saved midpoint binding")
        result = {"sha256": physical.design.run._write_new(out, value), "rows": len(value["rows"])}
    print(json.dumps(result), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    main()
