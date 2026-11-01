"""CPU-testable variable-bank CEM component; not a host or GPU launcher.

The old controller is reused with per-instance private function globals; its
module globals and all historical sources remain unchanged. An actual host
adapter must separately bind node budgets, identity and durable accounting.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import FunctionType

from . import codex_p1_full_cem_controller_20261101_exclusive as base
from . import codex_four_arm_full_cost_20261030 as spec

CAP = 195
ITERATIONS = 4
ELITES = 4
BOUNDARIES = (0, 5, 10, 15)
HORIZONS = (3, 2, 1, 1)
STRUCTURED_REPORTS = (
    "7aaa901a28974c09bd0cb12fb4b7a7efe939a533dbe2649327d9c33802fb5d9d",
    "4d4f4f074a1b3cdc0b9999c09c71414d78cd56825c067c139e1446f73e19aab2",
)


def allocate(cap, horizon):
    if type(cap) is not int or type(horizon) is not int or cap < 1 or horizon < 1:
        raise ValueError("positive integer cap/horizon required")
    samples = (cap // horizon - 1) // ITERATIONS
    if samples < ELITES:
        raise ValueError("cap insufficient for four elites and winner replay")
    return samples


def contract():
    samples = [allocate(CAP, h) for h in HORIZONS]
    return {
        "schema": "codex_p1_variable_cem_development_20261101.contract.v1",
        "claim": "development_only_not_confirmatory",
        "common_forward_cap": CAP, "boundaries": list(BOUNDARIES), "horizons": list(HORIZONS),
        "iterations": ITERATIONS, "elites": ELITES, "samples": samples,
        "forward_per_node": [(ITERATIONS*s+1)*h for s, h in zip(samples, HORIZONS)],
        "winner_replays_per_node": 1,
        "structured_observed_node_charges": [174, 192, 183],
        "structured_b15": "unknown_not_started", "equal_compute": False,
        "cost_scope": "forward_calls_only_not_backward_or_total_compute",
        "runtime_identity_authenticated": False,
        "host_integration_ready": False,
    }


def node_cost(boundary):
    if type(boundary) is not int or boundary not in BOUNDARIES:
        raise ValueError("invalid boundary")
    calls = contract()["forward_per_node"][BOUNDARIES.index(boundary)]
    cost = dict.fromkeys(spec.AXES, 0)
    cost.update(action_encode_calls=calls, predict_calls=calls, predict_samples=calls)
    return cost


def source_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def node_config(boundary, seed):
    node_cost(boundary)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("invalid seed")
    i = BOUNDARIES.index(boundary)
    return {"kind": "variable_cem", "seed": seed, "boundary": boundary,
            "horizon": HORIZONS[i], "iterations": ITERATIONS,
            "samples": contract()["samples"][i], "elites": ELITES,
            "common_forward_cap": CAP, "winner_replays": 1}


def node_contract(task, model_sha, hardware_sha, boundary, *, seed):
    """Construct a compatible leaf-driver contract, without creating a driver."""
    from . import codex_four_arm_common_suffix_control_20261030 as common
    _, value, _ = common.contract_for(task, model_sha, hardware_sha)
    value = deepcopy(value)
    config = node_config(boundary, seed)
    config_sha = spec.digest(config)
    optimizer = source_sha()
    value["identity"]["optimizer"] = optimizer
    value["identity"]["optimizer_config"] = config_sha
    value["phase_policy_sha256"] = spec.digest(config)
    for entry in value["arms"].values():
        entry.update(initializer_sha256=optimizer, optimizer_sha256=optimizer,
                     optimizer_config_sha256=config_sha)
    caps = node_cost(boundary)
    value["levels"] = [{"id": "node", "caps": caps},
                       {"id": "unused_larger", "caps": {k: v + 1 for k, v in caps.items()}}]
    spec.preflight(value, spec.digest(value))
    return value


def audit_development_sources():
    results = Path(__file__).resolve().parents[2] / "results"
    output = []
    for root, expected in enumerate(STRUCTURED_REPORTS):
        path = results / f"p1_first_feedback_carry_20261101_main_root{root}_RUN_MAIN02/report.json"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("structured report hash mismatch")
        report = json.loads(raw)
        nodes = report["nodes"][f"s{root}_structured"]
        if [n["boundary"] for n in nodes] != list(BOUNDARIES):
            raise ValueError("structured roster")
        charges = []
        unstarted = []
        for node in nodes:
            if node["status"] == "not_started":
                unstarted.append(node["boundary"])
                continue
            if node["status"] != "sealed":
                raise ValueError("unsealed structured node")
            receipt = node["receipt"]
            account = receipt["account"]
            counts = account["charged_upper_bounds"]
            hooks = receipt["module_hook_audit"]["module_forward_pre_hooks"]
            if (account["status"] != "closed_unverified"
                or counts["predict_calls"] != counts["action_encode_calls"]
                or hooks["predict_calls"] != counts["predict_calls"]
                or hooks["action_encode_calls"] != counts["action_encode_calls"]):
                raise ValueError("structured accounting mismatch")
            charges.append(counts["predict_calls"])
        if charges != [174, 192, 183] or unstarted != [15]:
            raise ValueError("structured observed charges drift")
        output.append({"root": root, "report_sha256": expected, "forward": charges,
                       "unstarted_boundaries": unstarted, "independent_confirmation": False})
    return output


class Controller(base.Controller):
    """CEM over isolated callbacks; only the separate host may authorize action."""

    def __init__(self, **kwargs):
        private = dict(base.Controller.step.__globals__)
        private.update(ITERATIONS=ITERATIONS, ELITES=ELITES)
        self._step_globals = private
        self._isolated_step = FunctionType(base.Controller.step.__code__, private,
                                           "isolated_variable_cem_step",
                                           base.Controller.step.__defaults__,
                                           base.Controller.step.__closure__)
        super().__init__(**kwargs)
        try:
            self.journal.append({"type": "variable_bank_contract", "contract": contract(),
                                 "precedence": "samples schedule supersedes base header samples",
                                 "host_authorization": False})
        except BaseException:
            self.close()
            raise

    def step(self):
        if self.stopped or self.index >= len(BOUNDARIES):
            raise RuntimeError("controller stopped or complete")
        self._step_globals["SAMPLES"] = contract()["samples"][self.index]
        return self._isolated_step(self)


if __name__ == "__main__":
    import json
    print(json.dumps(deepcopy(contract()), sort_keys=True, indent=2))
