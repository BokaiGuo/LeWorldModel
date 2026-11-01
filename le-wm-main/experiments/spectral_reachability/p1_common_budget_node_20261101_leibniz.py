"""Prospective common node-cap adapter for existing durable LeWM dispatch."""
from copy import deepcopy
import hashlib
from pathlib import Path

from . import codex_four_arm_full_cost_20261030 as spec
from . import codex_four_arm_durable_20261030 as durable
from . import codex_four_arm_leaf_suffix_20261030 as leafmod

ARMS = ("structured", "variable_cem")
BOUNDARIES = (0, 5, 10, 15)
IDENTITY_FIELDS = ("root", "model", "goal", "action_domain")
OPTIMIZER_FIELDS = ("optimizer", "optimizer_config")
CAPS = dict.fromkeys(spec.AXES, 0)
CAPS.update(action_encode_calls=195, predict_calls=195, predict_samples=195,
            backward_calls=240, vjp_rows=240, jacobian_calls=1, svd_calls=1,
            replay_calls=64, selection_evaluations=64)


class EpisodeOrder:
    """Fail-stop boundary and physical return state; no physical dispatch."""
    def __init__(self):
        self.steps = 0
        self.pending = False
        self.stopped = False

    def begin(self, boundary):
        if not (not self.stopped and not self.pending and self.steps < 20
                and type(boundary) is int and boundary == self.steps and boundary in BOUNDARIES):
            self.failed()
            raise ValueError("invalid boundary transition")
        self.pending = True

    def returned(self, steps, stopped):
        if (not self.pending or self.stopped or type(steps) is not int
                or not 1 <= steps <= 5 or type(stopped) is not bool
                or (steps < 5 and not stopped)):
            self.failed()
            raise ValueError("invalid packet receipt; fail stopped")
        self.steps += steps
        self.pending = False
        self.stopped = stopped or self.steps == 20

    def failed(self):
        self.pending, self.stopped = False, True


def check_sources(sources):
    spec.require(type(sources) is dict and bool(sources), "sources required")
    required = {str(Path(module.__file__).resolve()) for module in (spec, durable, leafmod)}
    required.add(str(Path(__file__).resolve()))
    spec.require(required <= set(sources), "meter source closure")
    for name, digest in sources.items():
        path = Path(name)
        spec.require(path.is_absolute() and not path.is_symlink() and path.is_file()
                     and spec.is_hash(digest), "source identity")
        spec.require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, "source drift")


def make_protocol(templates, sources):
    """Freeze identities and common caps before either arm starts.

    Templates provide node optimizer identities; task/history are allowed to
    change as actual observations arrive. This does not authenticate templates.
    """
    check_sources(sources)
    spec.require(type(templates) is dict and set(templates) == set(ARMS), "arm roster")
    nodes, common = {}, None
    for arm in ARMS:
        spec.require(type(templates[arm]) is dict and set(templates[arm]) == set(BOUNDARIES), "node roster")
        nodes[arm] = {}
        for boundary in BOUNDARIES:
            value = templates[arm][boundary]
            spec.preflight(value, spec.digest(value))
            identity = value["identity"]
            shared = {k: identity[k] for k in IDENTITY_FIELDS}
            if common is None:
                common = shared
            spec.require(shared == common, "root/model/goal/domain mismatch")
            nodes[arm][str(boundary)] = {k: identity[k] for k in OPTIMIZER_FIELDS}
    return {"schema": "p1_common_budget_node_20261101_leibniz.v1",
            "arms": list(ARMS), "boundaries": list(BOUNDARIES),
            "caps": deepcopy(CAPS), "identity": common, "node_optimizers": nodes,
            "sources": deepcopy(sources), "cap_scope": "per_node_dispatch_not_total_compute",
            "P1_admission": False, "independent_confirmation": False}


def bind(protocol, protocol_sha, arm, boundary, template):
    spec.require(spec.is_hash(protocol_sha) and spec.digest(protocol) == protocol_sha, "protocol digest")
    spec.require(protocol["schema"] == "p1_common_budget_node_20261101_leibniz.v1"
                 and protocol["arms"] == list(ARMS) and protocol["boundaries"] == list(BOUNDARIES)
                 and protocol["caps"] == CAPS and protocol["P1_admission"] is False
                 and protocol["independent_confirmation"] is False, "protocol policy")
    spec.require(arm in ARMS and type(boundary) is int and boundary in BOUNDARIES, "arm/boundary")
    check_sources(protocol["sources"])
    spec.preflight(template, spec.digest(template))
    identity = template["identity"]
    spec.require({k: identity[k] for k in IDENTITY_FIELDS} == protocol["identity"]
                 and {k: identity[k] for k in OPTIMIZER_FIELDS} ==
                 protocol["node_optimizers"][arm][str(boundary)], "runtime identity drift")
    contract = deepcopy(template)
    contract["levels"] = [{"id": "node", "caps": deepcopy(CAPS)},
                          {"id": "unused_larger", "caps": {k: v + 1 for k, v in CAPS.items()}}]
    spec.preflight(contract, spec.digest(contract))
    return contract


def open_node(directory, protocol, protocol_sha, arm, boundary, template, model, *,
              analytic_fixture=False):
    """Create a real durable driver and LeafAdapter using the bound cap.

    The caller owns lifecycle and must finish the driver, independently consume
    the journal, and prevent a planner bypassing this leaf.
    """
    contract = bind(protocol, protocol_sha, arm, boundary, template)
    contract_sha = spec.digest(contract)
    label = "structured" if arm == "structured" else "native"
    driver = durable.Driver(directory, contract, contract_sha, label, "node",
                            f"{arm}_b{boundary:02}")
    try:
        leaf = leafmod.LeafAdapter(model, driver, contract, contract_sha,
                                   analytic_fixture=analytic_fixture)
    except BaseException:
        try:
            driver.dispatch("setup", "__abort__", {}, lambda: None)
        except ValueError:
            pass
        finally:
            driver.finish()
        raise
    return driver, leaf, contract
