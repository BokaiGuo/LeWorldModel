"""Physically execute one fixed Adam update checkpoint on all 24 prefixes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_uphill_multistart_physical_20261030_main as prior
from . import p1_adam_checkpoint_curve_20261031 as curve

ROOT = prior.physical.ROOT
PLAN = ROOT / "results/p1_adam_checkpoint_curve_20261031_RUN01.json"
PLAN_SHA = "46d7126e0c0f5354f829fa0acac27e1639c1f8ef94ac74ebcca21cfda55ef2a7"
need, sha = prior.need, prior.sha


def convert(plan, step):
    need(step in (5, 10), "fixed new budget")
    need(plan["schema"] == curve.NAME and plan["status"] == "complete"
         and plan["P1_admission"] is False and plan["independent_confirmation"] is False
         and plan["physical_execution"] is False and len(plan["rows"]) == 24,
         "curve contract")
    rows = []
    for i, row in enumerate(plan["rows"]):
        checkpoint = row["checkpoints"][str(step)]
        need(row["index"] == i and len(checkpoint["scores"]) == len(checkpoint["candidate_actions"])
             == len(checkpoint["candidate_sha256"]) == 4, "candidate roster")
        winner = prior.planner.adam.prior.pick(checkpoint["scores"])
        need(winner == checkpoint["winner"], "winner")
        actions = [np.asarray(a, dtype="<f4") for a in checkpoint["candidate_actions"]]
        for action, digest in zip(actions, checkpoint["candidate_sha256"], strict=True):
            need(action.shape == (2, 5, 2) and np.isfinite(action).all()
                 and (np.abs(action) <= 1).all()
                 and hashlib.sha256(action.tobytes()).hexdigest() == digest, "candidate action")
        rows.append({"index": i, "role": row["role"], "logical_pair": row["logical_pair"],
                     "initial_f32_sha256": row["initial_sha256"],
                     "candidate_scores": checkpoint["scores"], "selected_index": winner,
                     "selected_actions": actions[winner].tolist(),
                     "selected_action_sha256": checkpoint["candidate_sha256"][winner]})
    return {"schema": prior.planner.NAME, "status": "complete", "P1_admission": False,
            "independent_confirmation": False, "rows": rows, "sources": plan["sources"]}


def configure(step):
    name = f"p1_adam_checkpoint_physical_{step}_20261031"

    def build():
        need(sha(PLAN) == PLAN_SHA and sha(prior.OLD) == prior.OLD_SHA
             and sha(prior.planner.bundle.OUT) == prior.planner.BUNDLE_SHA, "input hashes")
        plan = json.loads(PLAN.read_bytes())
        old = json.loads(prior.OLD.read_bytes())
        blind = json.loads(prior.planner.bundle.OUT.read_bytes())
        roster = prior.assemble(convert(plan, step), old, blind)
        bindings = dict(old["bindings"])
        for path, digest in plan["sources"].items():
            need(path not in bindings or bindings[path] == digest, "source collision")
            bindings[path] = digest
        for path in (PLAN, prior.OLD, Path(__file__).resolve(), Path(prior.__file__).resolve(),
                     Path(prior.physical.__file__).resolve()):
            digest = sha(path)
            need(str(path) not in bindings or bindings[str(path)] == digest, "binding collision")
            bindings[str(path)] = digest
        need(all(sha(path) == digest for path, digest in bindings.items()), "source drift")
        return {"schema": name + ".prepare", "roster": roster, "bindings": bindings,
                "plan_sha256": PLAN_SHA, "updates_per_start": step, "fixed_rows": 24,
                "P1_admission": False, "independent_confirmation": False,
                "fairness_proven": False, "os_isolation_certified": False}

    prior.physical.NAME = prior.physical.engine.NAME = name
    prior.physical.build = build
    return name, build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "consume"))
    parser.add_argument("--step", type=int, choices=(5, 10), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    parser.add_argument("--directory")
    parser.add_argument("--report-sha")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    name, build = configure(args.step)
    out = Path(args.output)
    if args.command == "prepare":
        need(out.is_absolute() and out.parent == ROOT / "results"
             and out.name.startswith(name + "_prepare_") and not out.exists(), "output")
        result = {"sha256": prior.physical.design.run._write_new(out, build())}
    elif args.command == "run":
        need(args.authorize_physics, "physics authorization")
        result = prior.physical.execute(args.prepared, args.sha256, args.output)
    else:
        need(out.is_absolute() and out.parent == ROOT / "results"
             and out.name.startswith(name + "_consume_") and not out.exists(), "output")
        value = prior.physical.consume(args.directory, args.report_sha, args.prepared, args.sha256)
        result = {"sha256": prior.physical.design.run._write_new(out, value), "rows": len(value["rows"])}
    print(json.dumps(result), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
