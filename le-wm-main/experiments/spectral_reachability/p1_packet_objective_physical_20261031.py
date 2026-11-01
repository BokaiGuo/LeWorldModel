"""Execute all fixed packet-objective suffixes with the sealed physical engine."""
import argparse
import json
from pathlib import Path

from . import p1_packet_objective_adam_20261031 as planner
from . import p1_uphill_multistart_physical_20261030_main as prior

physical = prior.physical
NAME = "p1_packet_objective_physical_20261031"
PLAN = physical.ROOT / "results/p1_packet_objective_adam_20261031_RUN01.json"
PLAN_SHA = "3614dad72f2a1ca38662c390e326a543bec927d512742ce8adf8aae422f987e1"
need, sha = physical.need, physical.sha


def assemble(plan, original, blind):
    need(plan["schema"] == planner.NAME and plan["status"] == "complete"
         and plan["model_before"] == plan["model_after"]
         and plan["config"]["objective"] == "0.5 * (goal_mse_at_5 + goal_mse_at_10)",
         "packet plan identity")
    for row in plan["rows"]:
        need(len(row["candidate_scores"]) == len(row["candidate_packet_scores"]) == 4,
             "candidate roster")
        for score, components in zip(row["candidate_scores"], row["candidate_packet_scores"], strict=True):
            need(len(components) == 2 and abs(score - 0.5 * sum(components)) < 1e-6,
                 "objective reconstruction")
    compatible = dict(plan, schema=prior.planner.NAME)
    return prior.assemble(compatible, original, blind)


def build():
    need(sha(PLAN) == PLAN_SHA and sha(prior.OLD) == prior.OLD_SHA
         and sha(prior.planner.bundle.OUT) == prior.planner.BUNDLE_SHA, "sealed inputs")
    plan = json.loads(PLAN.read_bytes())
    original = json.loads(prior.OLD.read_bytes())
    blind = json.loads(prior.planner.bundle.OUT.read_bytes())
    roster = assemble(plan, original, blind)
    bindings = dict(original["bindings"])
    for path, digest in plan["sources"].items():
        need(path not in bindings or bindings[path] == digest, "source collision")
        bindings[path] = digest
    for path in (PLAN, prior.OLD, Path(__file__).resolve(), Path(physical.__file__).resolve()):
        bindings[str(path)] = sha(path)
    need(all(sha(path) == digest for path, digest in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "roster": roster, "bindings": bindings,
            "plan_sha256": PLAN_SHA, "fixed_rows": 24, "P1_admission": False,
            "independent_confirmation": False, "fairness_proven": False,
            "os_isolation_certified": False}


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
    physical.NAME = NAME
    physical.engine.NAME = NAME
    physical.build = build
    out = Path(args.output)
    if args.command == "prepare":
        need(out.is_absolute() and out.parent == physical.ROOT / "results"
             and out.name.startswith(NAME + "_prepare_"), "output")
        result = {"sha256": physical.design.run._write_new(out, build())}
    elif args.command == "run":
        need(args.authorize_physics, "physics authorization")
        result = physical.execute(args.prepared, args.sha256, args.output)
    else:
        need(out.is_absolute() and out.parent == physical.ROOT / "results"
             and out.name.startswith(NAME + "_consume_"), "output")
        value = physical.consume(args.directory, args.report_sha, args.prepared, args.sha256)
        result = {"sha256": physical.design.run._write_new(out, value), "rows": len(value["rows"])}
    print(json.dumps(result), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    main()
