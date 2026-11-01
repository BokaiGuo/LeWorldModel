"""Physically replay fixed first packets and real-midpoint optimized second packets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_uphill_multistart_physical_20261030_main as original
from . import p1_uphill_midpoint_feedback_plan_20261031_main as planner

ROOT = original.ROOT if hasattr(original, "ROOT") else original.physical.ROOT
NAME = "p1_uphill_midpoint_feedback_physical_20261031_main"
PLAN = ROOT / "results/p1_uphill_midpoint_feedback_plan_20261031_main_RUN01.json"
PLAN_SHA = "8ac6af3059fe7ff70158fdc861f726ec9927f0c0dbc56bdc07c312b8b1ac066a"
OLD = ROOT / "results/p1_uphill_multistart_physical_20261030_main_prepare_RUN01.json"
OLD_SHA = "c58e71cf4a5f42d4ab5668993e95869d64718a3b3d5ed37172816dc980e82ff8"
need, sha = original.need, original.sha


def assemble(old, plan, inputs):
    need(plan["schema"] == planner.NAME and plan["status"] == "complete"
         and len(old["roster"]) == len(plan["rows"]) == len(inputs) == 24, "fixed roster")
    roster = []
    for i, (item, row, (blind, pair, raw, _, frame)) in enumerate(zip(
            old["roster"], plan["rows"], inputs, strict=True)):
        need(item["index"] == row["index"] == blind["index"] == pair["index"] == i
             and item["role"] == row["role"] == blind["role"] == pair["role"]
             and item["logical_pair"] == row["logical_pair"] == blind["logical_pair"]
             and row["midpoint_pixel_sha256"] == frame["pixel_sha256"], "identity")
        first, second = bytes.fromhex(row["first_packet_hex"]), bytes.fromhex(row["second_packet_hex"])
        expected_choice = "optimized" if row["optimized_score"] < row["baseline_score"] else "original"
        need(len(first) == len(second) == 40 and first == raw[0].tobytes()
             and first.hex() == item["raw10_float32_hex"][:80]
             and row["chosen"] == expected_choice
             and (row["chosen"] != "original" or second == raw[1].tobytes()), "packet or choice")
        packet = first + second
        values = np.frombuffer(packet, dtype="<f4")
        need(np.isfinite(values).all() and (np.abs(values) <= 1).all(), "packet bounds")
        roster.append({**item, "raw10_float32_hex": packet.hex(),
                       "raw10_sha256": hashlib.sha256(packet).hexdigest()})
    return roster


def build():
    need(sha(PLAN) == PLAN_SHA and sha(OLD) == OLD_SHA, "input hashes")
    old, plan = json.loads(OLD.read_bytes()), json.loads(PLAN.read_bytes())
    inputs, input_sources = planner.inputs()
    roster = assemble(old, plan, inputs)
    bindings = dict(old["bindings"])
    for mapping in (plan["sources"], input_sources):
        for path, digest in mapping.items():
            need(path not in bindings or bindings[path] == digest, "binding collision")
            bindings[path] = digest
    for path in (PLAN, OLD, Path(__file__).resolve()):
        bindings[str(path)] = sha(path)
    need(all(sha(path) == digest for path, digest in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "roster": roster, "bindings": bindings,
            "plan_sha256": PLAN_SHA, "fixed_rows": 24, "P1_admission": False,
            "independent_confirmation": False, "fairness_proven": False}


def configure():
    physical = original.physical
    physical.NAME = NAME
    physical.engine.NAME = NAME
    physical.build = build


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
    configure()
    physical, out = original.physical, Path(args.output)
    if args.command == "prepare":
        need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_prepare_"),
             "output")
        result = {"sha256": physical.design.run._write_new(out, build())}
    elif args.command == "run":
        need(args.authorize_physics, "physics authorization")
        result = physical.execute(args.prepared, args.sha256, args.output)
    else:
        need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_consume_"),
             "output")
        value = physical.consume(args.directory, args.report_sha, args.prepared, args.sha256)
        result = {"sha256": physical.design.run._write_new(out, value)}
    print(json.dumps(result), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
