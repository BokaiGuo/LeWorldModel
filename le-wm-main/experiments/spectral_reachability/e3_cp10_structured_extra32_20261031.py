"""Exploratory cp10 feedback with structured winner refinement enabled."""
import argparse
import importlib
import json
from pathlib import Path

from . import p1_four_arm_extra32_budget_20261031 as control

NAME = "e3_cp10_structured_extra32_20261031"
PROTOCOL = Path(__file__).resolve().parents[3] / "docs/D2P_E3_STRUCTURED_EXTRA32_PROTOCOL_20261031.md"


def setup(root):
    if root not in (0, 1):
        raise ValueError("root must be 0 or 1")
    experiment = importlib.import_module(
        f".e3_cp10_four_arm_root{root}_20261031", __package__
    )
    control.EXTRA_ARMS = ("structured", "native", "random")
    experiment.configure()
    runner = experiment.runner
    old_build, old_protocol = runner.build, runner.protocol

    def build():
        value = old_build()
        for path in (Path(__file__).resolve(), PROTOCOL):
            value["sources"][str(path)] = runner.sha(path)
        return value

    def protocol():
        value = old_protocol()
        value.update(exploratory_structured_extra32=True, independent_confirmation=False,
                     P1_admission=False, fairness_proven=False)
        return value

    runner.NAME = f"{NAME}_root{root}"
    runner.build, runner.protocol = build, protocol
    return experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, choices=(0, 1), required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    run = sub.add_parser("run")
    for key in ("prepared", "sha256", "output", "authorization"):
        run.add_argument("--" + key, required=True)
    check = sub.add_parser("check")
    check.add_argument("--directory", required=True)
    check.add_argument("--report-sha", required=True)
    args = parser.parse_args()
    experiment = setup(args.root)
    runner = experiment.runner
    if args.command == "prepare":
        result = runner.prepare(args.output)
    elif args.command == "run":
        with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
            result = runner.run(args.prepared, args.sha256, args.output, args.authorization)
    else:
        saved = importlib.import_module(
            f".e3_cp10_four_arm_root{args.root}_saved_20261031", __package__
        )
        result = saved.check(args.directory, args.report_sha)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "recorded_known_root_not_admitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
