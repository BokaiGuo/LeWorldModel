"""Prospective common node budget and CEM replay/selection dispatch adapter."""
import ast
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from types import FunctionType

import numpy as np
import torch

from . import p1_common_budget_node_20261101_leibniz as common
from . import codex_p1_variable_cem_20261101_exclusive as variable

spec = common.spec
base = variable.base
CAPS = deepcopy(common.CAPS)
CAPS.update(replay_calls=193, selection_evaluations=193)


def required_sources():
    return {str(Path(p).resolve()) for p in (
        __file__, common.__file__, variable.__file__, base.__file__,
        inspect.getsourcefile(base._normal))}


def protocol(base_protocol):
    common.check_sources(base_protocol["sources"])
    spec.require(required_sources() <= set(base_protocol["sources"]), "CEM source closure")
    spec.require(base_protocol["caps"] == common.CAPS, "base cap identity")
    return {"schema": "p1_common_budget_cem_events_20261101_leibniz.v1",
            "base_protocol": deepcopy(base_protocol), "caps": deepcopy(CAPS),
            "semantics": "each candidate and winner replay: replay marker then actual score selection",
            "P1_admission": False, "independent_confirmation": False}


def bind(value, digest, arm, boundary, template):
    spec.require(spec.is_hash(digest) and spec.digest(value) == digest, "protocol digest")
    spec.require(value == protocol(value["base_protocol"]), "protocol policy or source drift")
    c = common.bind(value["base_protocol"], spec.digest(value["base_protocol"]), arm, boundary, template)
    c["levels"] = [{"id": "node", "caps": deepcopy(CAPS)},
                   {"id": "unused_larger", "caps": {k: v + 1 for k, v in CAPS.items()}}]
    spec.preflight(c, spec.digest(c))
    return c


def _score(self, leaf, states, history, goal, mean, std, matrix, boundary, horizon, raw):
    """Mirror the original score, with non-nested write-ahead cost events."""
    driver = self.leaf.driver
    with self.journal.attempt("rollout", details={"boundary": boundary, "horizon": horizon}):
        reservation = dict.fromkeys(spec.AXES, 0)
        reservation["replay_calls"] = 1
        driver.dispatch("verification", "replay", reservation, lambda: None)
        plan = torch.as_tensor(raw, dtype=states.dtype, device=states.device).unsqueeze(0)
        with torch.no_grad():
            predicted = base._normal(leaf, "verification", states, history, plan, mean, std)
            def evaluate():
                residual = (predicted - goal).double()
                value = ((residual @ matrix.double().T).square().mean()).item()
                spec.require(np.isfinite(value), "nonfinite score")
                return float(value)
            reservation = dict.fromkeys(spec.AXES, 0)
            reservation["selection_evaluations"] = 1
            return driver.dispatch("verification", "selection", reservation, evaluate)


def _instrumented_step():
    """Clone the actual sealed controller step, replacing only its local score."""
    source = textwrap.dedent(inspect.getsource(base.Controller.step))
    tree = ast.parse(source)
    methods = tree.body
    spec.require(len(methods) == 1 and isinstance(methods[0], ast.FunctionDef)
                 and methods[0].name == "step", "unexpected controller source")
    functions = [x for x in ast.walk(methods[0])
                 if isinstance(x, ast.FunctionDef) and x.name == "score"]
    spec.require(len(functions) == 1 and [a.arg for a in functions[0].args.args] == ["raw"],
                 "score injection point")
    replacement = ast.parse(
        "def score(raw):\n"
        "    return _common_score(self, leaf, states, history, goal, mean, std, matrix, boundary, horizon, raw)\n"
    ).body[0]
    functions[0].body = replacement.body
    ast.fix_missing_locations(tree)
    namespace = dict(base.Controller.step.__globals__)
    namespace["_common_score"] = _score
    exec(compile(tree, str(Path(base.__file__).resolve()), "exec"), namespace)
    return namespace["step"]


class Controller(variable.Controller):
    """CEM controller using the actual common-cap durable leaf for each node."""
    def __init__(self, *, budget_protocol, budget_sha, **kwargs):
        spec.require(budget_protocol == protocol(budget_protocol["base_protocol"])
                     and spec.digest(budget_protocol) == budget_sha, "budget protocol")
        self.budget_protocol, self.budget_sha = deepcopy(budget_protocol), budget_sha
        super().__init__(**kwargs)
        function = _instrumented_step()
        self._step_globals = function.__globals__
        self._isolated_step = FunctionType(function.__code__, self._step_globals,
                                           "common_metered_cem_step",
                                           function.__defaults__, function.__closure__)

    def step(self):
        try:
            self._check_node()
            return super().step()
        except BaseException:
            self.stopped = True
            driver = getattr(self.leaf, "driver", None)
            if type(driver) is common.durable.Driver and not driver.failed and not driver.closed:
                try:
                    driver.dispatch("setup", "__abort__", {}, lambda: None)
                except ValueError:
                    pass
            raise

    def _check_node(self):
        spec.require(type(self.leaf) is common.leafmod.LeafAdapter, "durable leaf required")
        driver = self.leaf.driver
        spec.require(type(driver) is common.durable.Driver and driver.caps == CAPS
                     and not driver.failed and not driver.closed
                     and all(v == 0 for v in driver.charged.values()), "fresh common driver required")
        spec.require(self.budget_protocol == protocol(self.budget_protocol["base_protocol"])
                     and spec.digest(self.budget_protocol) == self.budget_sha, "protocol/source drift")
        spec.require(self.index < len(variable.BOUNDARIES), "boundary exhausted")
        with driver.path.open("rb") as stream:
            header = json.loads(stream.readline())["payload"]
        contract = header["contract"]
        spec.require(header["arm"] == "native" and header["level"] == "node"
                     and header["contract_sha256"] == self.leaf.contract_sha == spec.digest(contract)
                     and bind(self.budget_protocol, self.budget_sha, "variable_cem",
                              variable.BOUNDARIES[self.index], contract) == contract,
                     "node contract binding")
