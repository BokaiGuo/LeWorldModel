"""Known-root variable CEM host. Prepare is CPU-only; run needs fresh authorization."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
from types import FunctionType, SimpleNamespace
from unittest.mock import patch

from . import codex_p1_full_cem_host_20261101_exclusive as old
from . import codex_p1_full_cem_host_v2_20261101_exclusive as v2
from . import codex_p1_variable_cem_20261101_exclusive as variable
from .codex_e1_proposal_durable_events_20261101 import read_events

NAME = "codex_p1_variable_cem_host_20261101_exclusive"
AUTH = "MAIN_AUTHORIZES_P1_VARIABLE_CEM_DEVELOPMENT_CUDA_AND_PHYSICS"
HERE = Path(__file__).resolve()
DOC = HERE.parents[3] / "docs/D2P_P1_VARIABLE_CEM_HOST_20261101_EXCLUSIVE.md"
TEST = HERE.parents[2] / "tests/test_codex_p1_variable_cem_host_20261101_exclusive.py"


def node_cost(boundary):
    return variable.node_cost(boundary)


def node_contract(task, model_sha, hardware_sha, boundary):
    return variable.node_contract(task, model_sha, hardware_sha, boundary, seed=old.SEED)


def make_protocol(source, root):
    old.spec.require(type(root) is int and root in (0, 1)
                     and source.get("root_slot") == root, "root protocol mismatch")
    old.spec.require(source.get("boundaries") == list(variable.BOUNDARIES)
                     and source.get("feedback_horizons") == list(variable.HORIZONS), "schedule")
    result = deepcopy(source)
    result.update(schema=f"{NAME}_root{root}.protocol.v1",
                  node_costs={str(b): node_cost(b) for b in variable.BOUNDARIES},
                  node_caps_by_boundary={str(b): node_cost(b) for b in variable.BOUNDARIES},
                  variable_cem=variable.contract(), independent_confirmation=False,
                  fairness_proven=False, equal_compute=False, P1_admission=False,
                  full_cost_certified=False, claim="known_root_development_only")
    result.pop("cem", None)
    return result


def check_prepare_name(path, root):
    old.spec.require(type(root) is int and root in (0, 1), "invalid root")
    p = Path(path)
    old.spec.require(p.name.startswith(f"{NAME}_root{root}_PREP_") and p.name.endswith(".json"),
                     "root-specific variable prepare required")
    return p


def verify_node_events(path, boundary):
    """Verify the durable variable-bank schedule before dispatching a packet."""
    budget = node_cost(boundary)
    records = read_events(path)
    rows = [row["payload"]["event"] for row in records]
    contracts = [r for r in rows if r.get("type") == "variable_bank_contract"]
    old.spec.require(len(contracts) == 1 and contracts[0]["contract"] == variable.contract()
                     and len(rows) > 1 and rows[1] is contracts[0], "variable contract")
    selected = [r for r in rows if r.get("type") == "selection" and r.get("boundary") == boundary]
    replays = [r for r in rows if r.get("type") == "winner_replay" and r.get("boundary") == boundary]
    contexts = [i for i, r in enumerate(rows) if r.get("type") == "context" and r.get("boundary") == boundary]
    old.spec.require(len(contexts) == len(selected) == len(replays) == 1, "node context or selection")
    end = rows.index(replays[0])
    old.spec.require(contexts[0] < rows.index(selected[0]) < end, "node order")
    events = rows[contexts[0]:end + 1]
    count = variable.contract()["samples"][variable.BOUNDARIES.index(boundary)]
    horizon = variable.HORIZONS[variable.BOUNDARIES.index(boundary)]
    banks = [e for e in events if e.get("type") == "bank"]
    scores = [e for e in events if e.get("type") == "score"]
    expected = variable.ITERATIONS * count
    old.spec.require(len(banks) == variable.ITERATIONS and len(scores) == expected
                     and [e["index"] for e in scores] == list(range(expected)), "bank/score count")
    for i, bank in enumerate(banks):
        old.spec.require(bank["boundary"] == boundary and bank["iteration"] == i
                         and len(bank["raw"]) == count, "bank size/order")
        old.spec.require(all(len(plan) == horizon for plan in bank["raw"]), "bank horizon")
    winner = min(range(expected), key=lambda i: (scores[i]["value"], i))
    pick = selected[0]
    old.spec.require(all(math.isfinite(s["value"]) for s in scores)
                     and pick["index"] == winner
                     and pick["raw"] == banks[winner // count]["raw"][winner % count]
                     and pick["score"] == scores[winner]["value"]
                     and math.isclose(replays[0]["score"], pick["score"], rel_tol=1e-6, abs_tol=1e-8),
                     "winner/replay mismatch")
    attempts = [(i, r) for i, r in enumerate(rows) if contexts[0] <= i <= end
                and r.get("type") == "attempt"]
    returns = [r.get("attempt_index") for r in events if r.get("type") == "return"]
    for operation, number in (("rollout", expected + 1),
                              ("action_encode", budget["action_encode_calls"]),
                              ("predict", budget["predict_calls"])):
        indexes = [i for i, r in attempts if r.get("operation") == operation]
        old.spec.require(len(indexes) == number and all(returns.count(i) == 1 for i in indexes),
                         f"incomplete {operation} accounting")
        if operation == "rollout":
            old.spec.require(all(rows[i]["details"] == {"boundary": boundary, "horizon": horizon}
                                 for i in indexes), "rollout identity mismatch")
    old.spec.require(not any(r.get("type") == "failure" for r in events), "failed event in node")
    old.spec.require(budget["predict_calls"] == (expected + 1) * horizon <= variable.CAP,
                     "node budget mismatch")


def isolated_execute():
    """Clone the existing driver with private globals; do not patch its module."""
    env = dict(old.execute_one.__globals__)
    env.update(NAME=NAME, node_cost=node_cost, node_contract=node_contract,
               verify_node_events=verify_node_events,
               cem=SimpleNamespace(**{**vars(old.cem), "Controller": variable.Controller}))
    return FunctionType(old.execute_one.__code__, env, "variable_execute_one",
                        old.execute_one.__defaults__, old.execute_one.__closure__)


@contextmanager
def configured(root):
    with v2.configured(root) as runner:
        parent_build, parent_protocol = runner.build, runner.protocol
        execute = isolated_execute()
        target = (old.root0, old.root1)[root]

        def protocol():
            return make_protocol(parent_protocol(), root)

        def build():
            value = parent_build()
            value["schema"] = f"{NAME}_root{root}.prepare.v1"
            value["protocol"] = protocol()
            for path in (HERE, Path(variable.__file__).resolve(), DOC, TEST):
                value["sources"][str(path)] = runner.sha(path)
            return value

        def execute_bound(prepared, data, store, visual, runtime, factory):
            return execute(prepared, data, store, visual, runtime, factory, runner, target.rebind, root)

        with patch.object(runner, "NAME", f"{NAME}_root{root}"), \
             patch.object(runner, "AUTH", AUTH), patch.object(runner, "protocol", protocol), \
             patch.object(runner, "build", build), patch.object(runner, "execute", execute_bound):
            yield runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, choices=(0, 1), required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-prepare")
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", required=True)
    run = sub.add_parser("run")
    for key in ("prepared", "sha256", "output", "authorization"):
        run.add_argument("--" + key, required=True)
    run.add_argument("--authorize-host-cuda-and-physics", action="store_true", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        check_prepare_name(args.output, args.root)
    elif args.command == "run":
        check_prepare_name(args.prepared, args.root)
        old.spec.require(Path(args.output).name.startswith(f"{NAME}_root{args.root}_RUN_")
                         and args.authorization == AUTH, "output/authorization")
    with configured(args.root) as runner:
        if args.command == "dry-prepare":
            value = runner.build()
            task = value["source"]["tasks"][args.root]
            hashes = {}
            for boundary in variable.BOUNDARIES:
                node = node_contract(task, "a" * 64, "b" * 64, boundary)
                old.spec.require(node["levels"][0]["caps"] == value["protocol"]["node_costs"][str(boundary)],
                                 "dry node contract budget")
                hashes[str(boundary)] = old.spec.digest(node)
            result = {"status": "dry_prepared_no_model_or_physics", "root": args.root,
                      "digest": runner.digest(value), "protocol": value["protocol"],
                      "dry_node_contract_sha256": hashes}
        elif args.command == "prepare":
            result = runner.prepare(args.output)
        else:
            with old.gpu.locking.gpu_lock(old.gpu.LOCK):
                result = runner.run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "recorded_known_root_not_admitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
