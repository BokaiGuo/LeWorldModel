"""Physical execution of the matched imagined-midpoint control."""
import argparse
import json
from pathlib import Path

from . import p1_uphill_midpoint_feedback_physical_20261031_main as prior
from . import p1_uphill_imagined_midpoint_plan_20261031_main as planner

ROOT = prior.ROOT
NAME = "p1_uphill_imagined_midpoint_physical_20261031_main"
PLAN = ROOT / "results/p1_uphill_imagined_midpoint_plan_20261031_main_RUN01.json"
PLAN_SHA = "02e6f2fdf66df0d1f2f1bd6743f592dea820edc1d941d343ffecf7b853d6d300"
need = prior.need


def configure():
    prior.PLAN, prior.PLAN_SHA, prior.NAME = PLAN, PLAN_SHA, NAME
    prior.planner.NAME = planner.NAME
    prior.configure()
    physical = prior.original.physical

    def build():
        result = prior.build()
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
