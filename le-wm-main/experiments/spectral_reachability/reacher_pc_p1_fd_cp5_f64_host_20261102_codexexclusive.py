"""Versioned host wiring for float64 prefix parameters; no execution on import."""
import argparse
import json
from pathlib import Path

from . import reacher_pc_p1_fd_cp5_host_20261102_codexexclusive as host
from . import reacher_pc_p1_fd_cp5_f64_20261102_codexexclusive as bridge
from . import reacher_pc_p1_fd_cp5_saved_20261102_codexexclusive as saved

BASE_PLAN = host.plan


def plan(*paths):
    result = BASE_PLAN(*paths)
    result["schema"] = "fd_cp5_host_f64_v2"
    for file in (Path(__file__).resolve(), Path(bridge.original.__file__).resolve(),
                 Path(saved.__file__).resolve()):
        result["files"][str(file)] = bridge.sha_file(file)
    return result


def configure():
    host.bridge = bridge
    host.plan = plan


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    for key in ("model", "audit", "physical-plan", "preflight", "output"):
        prep.add_argument("--" + key, required=True)
    run = sub.add_parser("run")
    for key in ("plan", "sha256", "output", "authority"):
        run.add_argument("--" + key, required=True)
    consume = sub.add_parser("consume")
    for key in ("plan", "plan-sha", "journal", "journal-sha", "output"):
        consume.add_argument("--" + key, required=True)
    args = parser.parse_args()
    configure()
    if args.action == "prepare":
        host.write_new(args.output, plan(args.model, args.audit, args.physical_plan, args.preflight))
        print(json.dumps({"plan": args.output, "sha256": bridge.sha_file(args.output)}))
    elif args.action == "run":
        print(json.dumps(host.run(args.plan, args.sha256, args.output, args.authority)))
    else:
        result = saved.consume(args.plan, args.plan_sha, args.journal, args.journal_sha)
        host.write_new(args.output, result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
