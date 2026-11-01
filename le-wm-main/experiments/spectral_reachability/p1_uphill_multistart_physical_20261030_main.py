"""Execute fixed uphill multistart suffixes with the existing physical verifier."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_uphill_cp5_physical20_20261030 as physical
from . import p1_uphill_multistart_adam_20261030_main as planner

NAME = "p1_uphill_multistart_physical_20261030_main"
PLAN = physical.ROOT / "results/p1_uphill_multistart_adam_20261030_main_RUN01.json"
PLAN_SHA = "018dffcb53e5e4dd40e6323fc3665ca53321cb027023c0cc4c1edb5daa06229f"
OLD = physical.ROOT / "results/p1_uphill_cp5_physical20_20261030_prepare_RUN01.json"
OLD_SHA = "7984bcd309a008ed63a4d0752be5726361b30e5042eb129cc42d72f7be1e42b9"
need, sha = physical.need, physical.sha


def assemble(plan, prior, blind):
    need(plan["schema"] == planner.NAME and plan["status"] == "complete"
         and plan["P1_admission"] is False and plan["independent_confirmation"] is False
         and len(plan["rows"]) == len(prior["roster"]) == len(blind["rows"]) == 24,
         "complete fixed roster")
    need(plan["sources"][str(planner.bundle.OUT)] == planner.BUNDLE_SHA, "plan bundle binding")
    roster = []
    for i, (row, old, inp) in enumerate(zip(plan["rows"], prior["roster"], blind["rows"], strict=True)):
        need(row["index"] == old["index"] == inp["index"] == i
             and row["role"] == old["role"] == inp["role"]
             and row["logical_pair"] == old["logical_pair"] == inp["logical_pair"]
             and old["prefix5_float32_hex"] == inp["prefix_raw_f32_hex"]
             and old["cp5_pixel_sha256"] == inp["cp5_pixel_sha256"], "prefix and image identity")
        starts = planner.adam.prior.initial_actions(row["logical_pair"])
        need(hashlib.sha256(starts.tobytes()).hexdigest() == row["initial_f32_sha256"]
             and planner.adam.prior.pick(row["candidate_scores"]) == row["selected_index"], "start and winner")
        raw = np.asarray(row["selected_actions"], dtype="<f4")
        need(raw.shape == (2, 5, 2) and np.isfinite(raw).all() and (np.abs(raw) <= 1).all()
             and hashlib.sha256(raw.tobytes()).hexdigest() == row["selected_action_sha256"], "action identity")
        item = dict(old)
        item.update(raw10_float32_hex=raw.tobytes().hex(), raw10_sha256=row["selected_action_sha256"],
                    plan_row_sha256=hashlib.sha256(json.dumps(
                        row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest())
        roster.append(item)
    return roster


def build():
    need(sha(PLAN) == PLAN_SHA and sha(OLD) == OLD_SHA
         and sha(planner.bundle.OUT) == planner.BUNDLE_SHA, "input hashes")
    plan = json.loads(PLAN.read_bytes())
    prior = json.loads(OLD.read_bytes())
    blind = json.loads(planner.bundle.OUT.read_bytes())
    roster = assemble(plan, prior, blind)
    bindings = dict(prior["bindings"])
    for path, digest in plan["sources"].items():
        need(path not in bindings or bindings[path] == digest, "source collision")
        bindings[path] = digest
    for path in (PLAN, OLD, Path(__file__).resolve(), Path(physical.__file__).resolve()):
        bindings[str(path)] = sha(path)
    need(all(sha(path) == digest for path, digest in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "roster": roster, "bindings": bindings,
            "plan_sha256": PLAN_SHA, "fixed_rows": 24, "P1_admission": False,
            "independent_confirmation": False, "fairness_proven": False, "os_isolation_certified": False}


def configure():
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
    output = Path(args.output)
    if args.command == "prepare":
        need(output.is_absolute() and output.parent == physical.ROOT / "results"
             and output.name.startswith(NAME + "_prepare_"), "output")
        result = {"sha256": physical.design.run._write_new(output, build())}
    elif args.command == "run":
        need(args.authorize_physics, "physics authorization")
        result = physical.execute(args.prepared, args.sha256, args.output)
    else:
        need(output.is_absolute() and output.parent == physical.ROOT / "results"
             and output.name.startswith(NAME + "_consume_"), "output")
        result = {"sha256": physical.design.run._write_new(
            output, physical.consume(args.directory, args.report_sha, args.prepared, args.sha256))}
    print(json.dumps(result), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
