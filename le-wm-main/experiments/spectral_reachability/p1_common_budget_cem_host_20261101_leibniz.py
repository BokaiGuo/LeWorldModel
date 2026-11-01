"""Known-root host: real RGB/physics with CEM replay/selection node accounting."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace
from unittest.mock import patch

from . import codex_p1_variable_cem_host_v2_20261101_exclusive as parent
from . import p1_common_budget_cem_events_20261101_leibniz as budget

previous = parent.previous
old, variable, spec = previous.old, previous.variable, previous.old.spec
NAME = "p1_common_budget_cem_host_20261101_leibniz"
AUTH = "AUTHORIZES_P1_COMMON_BUDGET_CEM_KNOWN_ROOT_CUDA_PHYSICS"
HERE = Path(__file__).resolve()
DOC = HERE.parents[3] / "docs/P1_COMMON_BUDGET_CEM_HOST_20261101_LEIBNIZ.md"
TEST = HERE.parents[2] / "tests/test_p1_common_budget_cem_host_20261101_leibniz.py"


def node_cost(boundary):
    result = variable.node_cost(boundary)
    index = variable.BOUNDARIES.index(boundary)
    n = variable.ITERATIONS * variable.contract()["samples"][index] + 1
    result.update(replay_calls=n, selection_evaluations=n)
    return result


def node_contract(task, model_sha, hardware_sha, boundary):
    result = variable.node_contract(task, model_sha, hardware_sha, boundary, seed=old.SEED)
    result["levels"] = [{"id": "node", "caps": deepcopy(budget.CAPS)},
                        {"id": "unused_larger", "caps": {k: v + 1 for k, v in budget.CAPS.items()}}]
    spec.preflight(result, spec.digest(result))
    return result


class LiveLeaf(old.LiveLeaf):
    @property
    def driver(self):
        spec.require(self.node is not None, "actual RGB node not active")
        return self.node.driver


def checked_score(self, leaf, states, history, goal, mean, std, matrix, boundary, horizon, raw):
    node = self.leaf.node
    spec.require(type(node) is old.leafmod.LeafAdapter, "actual metered node required")
    driver = node.driver
    spec.require(type(driver) is old.durable.Driver and not driver.closed and not driver.failed
                 and driver.caps == budget.CAPS, "common node driver")
    with driver.path.open("rb") as stream:
        header = json.loads(stream.readline())["payload"]
    contract = header["contract"]
    expected = variable.node_config(boundary, old.SEED)
    spec.require(header["arm"] == "native" and header["level"] == "node"
                 and header["contract_sha256"] == node.contract_sha == spec.digest(contract)
                 and contract["identity"]["optimizer"] == variable.source_sha()
                 and contract["identity"]["optimizer_config"] == spec.digest(expected)
                 and contract["levels"][0]["caps"] == budget.CAPS,
                 "score contract/source identity")
    return budget._score(self, leaf, states, history, goal, mean, std, matrix, boundary, horizon, raw)


class Controller(variable.Controller):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        function = budget._instrumented_step()
        self._step_globals = function.__globals__
        self._step_globals["_common_score"] = checked_score
        self._isolated_step = FunctionType(function.__code__, self._step_globals, "host_common_cem_step",
                                           function.__defaults__, function.__closure__)


def make_protocol(value, root):
    value = deepcopy(value)
    value.update(schema=f"{NAME}_root{root}.protocol.v1",
                 node_costs={str(b): node_cost(b) for b in variable.BOUNDARIES},
                 node_caps_by_boundary={str(b): deepcopy(budget.CAPS) for b in variable.BOUNDARIES},
                 common_node_caps=deepcopy(budget.CAPS),
                 score_accounting="replay before rollout; selection wraps actual scoring, including winner replay",
                 shared_upstream_cost="unknown_unallocated", P1_admission=False,
                 independent_confirmation=False, equal_compute=False, fairness_proven=False)
    return value


@contextmanager
def configured(root):
    with parent.configured(root) as runner:
        parent_build, parent_protocol = runner.build, runner.protocol

        def protocol():
            return make_protocol(parent_protocol(), root)

        def build():
            value = parent_build()
            value["schema"] = f"{NAME}_root{root}.prepare.v1"
            value["protocol"] = protocol()
            for path in (HERE, DOC, TEST, Path(budget.__file__).resolve(),
                         Path(budget.common.__file__).resolve()):
                spec.require(path.is_file() and not path.is_symlink(), "required adapter source")
                value["sources"][str(path)] = runner.sha(path)
            return value

        def execute(prepared, data, store, visual, runtime, factory):
            captured = []

            class TrackedController(Controller):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    captured.append(self)

            env = dict(old.execute_one.__globals__)
            env.update(NAME=NAME, node_cost=node_cost, node_contract=node_contract,
                       verify_node_events=previous.verify_node_events, LiveLeaf=LiveLeaf,
                       cem=SimpleNamespace(**{**vars(old.cem), "Controller": TrackedController}))
            fn = FunctionType(old.execute_one.__code__, env, "common_cem_execute_one",
                              old.execute_one.__defaults__, old.execute_one.__closure__)
            target = (old.root0, old.root1)[root]
            report = fn(prepared, data, store, visual, runtime, factory, runner, target.rebind, root)
            return parent.finalize_report(report, captured, store.path)

        with patch.object(runner, "NAME", f"{NAME}_root{root}"), \
             patch.object(runner, "AUTH", AUTH), patch.object(runner, "protocol", protocol), \
             patch.object(runner, "build", build), patch.object(runner, "execute", execute):
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
    prefix = f"{NAME}_root{args.root}"
    if args.command in ("prepare", "run"):
        path = Path(args.output if args.command == "prepare" else args.prepared)
        spec.require(path.is_absolute() and path.name.startswith(prefix + "_PREP_")
                     and path.name.endswith(".json"), "prepare name")
    if args.command == "run":
        spec.require(Path(args.output).is_absolute() and Path(args.output).name.startswith(prefix + "_RUN_")
                     and args.authorization == AUTH, "output/authorization")
    with configured(args.root) as runner:
        if args.command == "dry-prepare":
            value = runner.build()
            result = {"status": "dry_prepared_no_model_or_physics", "digest": runner.digest(value),
                      "root": args.root, "protocol": value["protocol"]}
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
