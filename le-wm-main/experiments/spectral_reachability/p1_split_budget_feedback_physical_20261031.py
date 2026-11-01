"""Guard and execute the fixed split-budget real-midpoint plans."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_split_budget_feedback_plan_20261031 as planner
from . import p1_packet_midpoint_feedback_physical_20261031 as guarded
from . import p1_uphill_packet_fidelity_20261031 as frames

ROOT = planner.ROOT
physical = guarded.physical
need, sha = physical.need, physical.sha
NAME = "p1_split_budget_feedback_physical_20261031"
PLAN = ROOT / "results/p1_split_budget_feedback_plan_20261031_RUN01.json"
PLAN_SHA = "15acce6919430688919ea76049555f95a9383f54531fd2f4b703c9d3071da335"
OLD = ROOT / "results/p1_adam_checkpoint_physical_10_20261031_prepare_RUN01.json"
OLD_SHA = "5be37dd41e7edfb228390a8924ecfaf9688ea0e54cd93386a3d042aaf43b9202"


def build():
    need(sha(PLAN) == PLAN_SHA and sha(OLD) == OLD_SHA, "input hashes")
    plan, old = json.loads(PLAN.read_bytes()), json.loads(OLD.read_bytes())
    need(plan["schema"] == planner.NAME and plan["status"] == "complete"
         and plan["model_before"] == plan["model_after"]
         and len(plan["rows"]) == len(old["roster"]) == 24, "fixed plan")
    all_inputs, stopped, sources = planner.midpoint.inputs()
    inputs = [x for x in all_inputs if x[0]["step"] == 10]
    need(len(inputs) == 24 and not any(x["step"] == 10 for x in stopped), "fixed inputs")
    terminal = planner.midpoint.terminal
    directory, run_sha = terminal.RUNS[10]
    report = terminal.pair.pairing.read(directory / "run_report.json", run_sha)
    prior = terminal.curve.planner.adam.prior
    roster = []
    for i, (row, item, (entry, _, pixels, pixel_sha)) in enumerate(
            zip(plan["rows"], old["roster"], inputs, strict=True)):
        raw = entry["action"]
        first, second = bytes.fromhex(row["first_packet_hex"]), bytes.fromhex(row["second_packet_hex"])
        need(i == row["index"] == item["index"] == entry["index"]
             and row["role"] == item["role"] == entry["row"]["role"]
             and row["logical_pair"] == item["logical_pair"] == entry["row"]["logical_pair"]
             and row["midpoint_pixel_sha256"] == pixel_sha
             and len(first) == len(second) == 40 and first == raw[0].tobytes()
             and first.hex() == item["raw10_float32_hex"][:80], "identity and first packet")
        starts = np.asarray(prior.initial_actions(row["logical_pair"])[:, 1], dtype="<f4")
        need(hashlib.sha256(starts.tobytes()).hexdigest() == row["initial_sha256"]
             and prior.pick(row["scores"]) == row["winner"], "initialization and winner")
        values = np.frombuffer(first + second, dtype="<f4")
        need(np.isfinite(values).all() and (np.abs(values) <= 1).all(), "action bounds")
        folder = directory / f"row_{i:02d}"
        saved = terminal.pair.pairing.read(folder / "report.json", report["shards"][i]["report_sha256"])
        frame = frames.frame_at(saved["row"], 5, item["condition"])
        need(frame["pixel_sha256"] == pixel_sha == hashlib.sha256(pixels.tobytes()).hexdigest(),
             "midpoint binding")
        roster.append({**item, "raw10_float32_hex": (first + second).hex(),
                       "raw10_sha256": hashlib.sha256(first + second).hexdigest(),
                       "midpoint": {k: frame[k] for k in ("pixel_sha256", "state", "snapshot")}})
    bindings = dict(old["bindings"])
    for mapping in (plan["sources"], sources):
        for path, digest in mapping.items():
            need(path not in bindings or bindings[path] == digest, "source collision")
            bindings[path] = digest
    for path in (PLAN, OLD, Path(__file__).resolve()):
        bindings[str(path)] = sha(path)
    need(all(sha(p) == h for p, h in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "roster": roster, "bindings": bindings,
            "plan_sha256": PLAN_SHA, "fixed_rows": 24, "P1_admission": False,
            "independent_confirmation": False, "fairness_proven": False}


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
        need(args.authorize_physics, "physics authorization")
        original_execute = physical.engine.execute
        physical.engine.execute = lambda *a: guarded.guarded(original_execute, *a)
        try:
            result = physical.execute(args.prepared, args.sha256, args.output)
        finally:
            physical.engine.execute = original_execute
    else:
        need(out.is_absolute() and out.parent == ROOT / "results"
             and out.name.startswith(NAME + "_consume_"), "output")
        value = physical.consume(args.directory, args.report_sha, args.prepared, args.sha256)
        result = {"sha256": physical.design.run._write_new(out, value), "rows": len(value["rows"])}
    print(json.dumps(result), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
