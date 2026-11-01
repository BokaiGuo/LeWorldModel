"""Prospective P1 total-budget host for structured and CEM planners.

This is a new opt-in entry point. It does not modify sealed runners or results.
The host freezes a complete resource reservation before model loading, routes
declared planner and physical operations through durable guards, and reports
uncovered bypass classes explicitly.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

from . import p1_physical_callback_guard_20261101_leibniz as physical
from . import p1_prospective_total_guard_20261101_leibniz as total

NAME = "p1_total_budget_host_20261101_codex"
PLANNERS = ("structured", "cem", "current_p1")
ROUTES = (
    "model_load", "source_encode", "action_encode", "predict", "backward",
    "jacobian", "svd", "solve", "candidate", "replay", "feedback",
    "physical",
)
ALGORITHM_KINDS = {
    "jacobian": "jacobian",
    "svd": "svd",
    "solve": "solve",
    "candidate": "candidate",
    "replay": "replay",
    "feedback": "feedback",
}
UNVERIFIED_BYPASSES = (
    "callbacks_can_invoke_uninstrumented_python_or_cuda_outside_the_host_api",
    "torch_functional_autograd_and_custom_cuda_kernels_are_not_globally_intercepted",
    "environment_calls_outside_physical_guard_are_not_process_sandboxed",
    "wall_time_flops_energy_and_gpu_contention_are_not_certified",
    "external_source_precommitment_time_and_root_independence_are_not_authenticated",
)
CONDITIONAL_AXES = ("render_calls", "reset_calls")


def _sha(path):
    path = Path(path).resolve()
    total.require(path.is_file() and not path.is_symlink(), "regular source file required")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required_sources():
    modules = (total, physical, physical.legacy, physical.moving)
    return tuple(sorted({str(Path(__file__).resolve())}
                        | {str(Path(module.__file__).resolve()) for module in modules}))


def build_protocol(*, run_id, planner, identity, reservation, physical_caps,
                   source_paths=(), conditional_axes=()):
    """Build an externally hashable protocol without executing model or physics."""
    sources = set(required_sources())
    sources.update(str(Path(path).resolve()) for path in source_paths)
    value = {
        "schema": NAME + ".protocol.v1",
        "run_id": run_id,
        "planner": planner,
        "identity": deepcopy(identity),
        "reservation": deepcopy(reservation),
        "physical_caps": deepcopy(physical_caps),
        "source_sha256": {path: _sha(path) for path in sorted(sources)},
        "routes": list(ROUTES),
        "conditional_axes": sorted(conditional_axes),
        "reservation_scope": "per_root_per_arm_start_before_load",
        "failure_policy": "attempts_charged_no_refund_fail_stop",
        "shared_initialization": "full_standalone_charge_each_arm",
        "unverified_bypasses": list(UNVERIFIED_BYPASSES),
        "P1_admission": False,
        "fairness_proven": False,
        "independent_confirmation": False,
    }
    validate_protocol(value, total.digest(value))
    return value


def validate_protocol(protocol, expected_sha):
    fields = {
        "schema", "run_id", "planner", "identity", "reservation",
        "physical_caps", "source_sha256", "routes", "conditional_axes",
        "reservation_scope", "failure_policy", "shared_initialization",
        "unverified_bypasses", "P1_admission", "fairness_proven",
        "independent_confirmation",
    }
    total.require(type(protocol) is dict and set(protocol) == fields, "protocol fields")
    total.require(protocol["schema"] == NAME + ".protocol.v1"
                  and type(protocol["run_id"]) is str and bool(protocol["run_id"])
                  and protocol["planner"] in PLANNERS, "protocol identity")
    total.vector(protocol["reservation"])
    physical._vector(protocol["physical_caps"])
    total.require(type(protocol["identity"]) is dict
                  and set(protocol["identity"]) == set(total.IDENTITY)
                  and all(type(value) is str and len(value) == 64
                          and all(char in "0123456789abcdef" for char in value)
                          for value in protocol["identity"].values()), "identity hashes")
    total.require(protocol["routes"] == list(ROUTES)
                  and protocol["reservation_scope"] == "per_root_per_arm_start_before_load"
                  and protocol["failure_policy"] == "attempts_charged_no_refund_fail_stop"
                  and protocol["shared_initialization"] == "full_standalone_charge_each_arm",
                  "routing or reservation policy")
    total.require(protocol["unverified_bypasses"] == list(UNVERIFIED_BYPASSES)
                  and protocol["P1_admission"] is False
                  and protocol["fairness_proven"] is False
                  and protocol["independent_confirmation"] is False,
                  "claim policy")
    conditional = protocol["conditional_axes"]
    total.require(type(conditional) is list and conditional == sorted(set(conditional))
                  and set(conditional) <= set(CONDITIONAL_AXES), "conditional axes")
    sources = protocol["source_sha256"]
    total.require(type(sources) is dict and set(required_sources()) <= set(sources),
                  "source closure")
    for name, digest in sources.items():
        path = Path(name)
        total.require(path.is_absolute() and path.is_file() and not path.is_symlink()
                      and type(digest) is str and len(digest) == 64
                      and _sha(path) == digest, "source identity")
    reservation = protocol["reservation"]
    physical_caps = protocol["physical_caps"]
    mapped_physical = {
        "reset_calls": physical_caps["reset"],
        "render_calls": physical_caps["goal_render"] + physical_caps["explicit_render"],
        "physical_dispatch_calls": physical_caps["public_step"],
        "physics_steps": physical_caps["physics_step"],
    }
    total.require(all(reservation[key] == value for key, value in mapped_physical.items()),
                  "physical and total reservations disagree")
    required_positive = {
        "model_load", "source_encode_calls", "source_encode_samples",
        "action_encode_calls", "predict_calls", "predict_samples",
        "candidate_evaluations", "physical_dispatch_calls", "physics_steps",
    }
    if protocol["planner"] == "structured":
        required_positive |= {
            "backward_calls", "vjp_rows", "jacobian_calls", "svd_calls",
            "solve_calls",
        }
    total.require(all(reservation[key] > 0 for key in required_positive),
                  "planner reservation omits required resource")
    total.require(total.digest(protocol) == expected_sha, "external protocol hash")
    return True


def total_contract(protocol):
    """Translate the outer protocol into the unchanged durable total guard."""
    validate_protocol(protocol, total.digest(protocol))
    return {
        "schema": total.NAME + ".contract.v1",
        "run_id": protocol["run_id"],
        "arm": "structured" if protocol["planner"] == "structured" else "cem",
        "caps": deepcopy(protocol["reservation"]),
        "source_sha256": deepcopy(protocol["source_sha256"]),
        "identity": deepcopy(protocol["identity"]),
        "shared_initialization": protocol["shared_initialization"],
        "hook_modules": deepcopy(total.HOOKS),
        "scope": protocol["reservation_scope"],
    }


class Host:
    """Single-use host. All public operations reserve before callback entry."""

    def __init__(self, directory, protocol, expected_sha):
        validate_protocol(protocol, expected_sha)
        directory = Path(directory)
        total.require(directory.is_absolute() and directory.is_dir(), "existing absolute output directory")
        contract = total_contract(protocol)
        contract_sha = total.digest(contract)
        self.protocol = deepcopy(protocol)
        self.protocol_sha = expected_sha
        self.total = total.Guard(directory / "total.jsonl", contract, contract_sha)
        physical_contract = {
            "schema": physical.NAME + ".contract.v1",
            "total_contract_sha256": contract_sha,
            "caps": deepcopy(protocol["physical_caps"]),
        }
        self.physical = physical.PhysicalGuard(
            directory / "physical.jsonl", self.total, protocol["physical_caps"],
            total.digest(physical_contract))
        self.model_loaded = False
        self.closed = False

    def load_model(self, loader):
        total.require(not self.model_loaded and callable(loader), "single model loader")
        model = self.total.dispatch("shared_setup", "model_load", 1, loader)
        self.total.install_model_sentinels(model)
        self.model_loaded = True
        return model

    def source_encode(self, samples, callback, *, phase="feedback"):
        total.require(self.model_loaded, "model must be loaded before encoding")
        return self.total.dispatch(phase, "source_encode", samples, callback)

    def action_encode(self, callback, *, phase="planning"):
        total.require(self.model_loaded, "model must be loaded before action encoding")
        return self.total.dispatch(phase, "action_encode", 1, callback)

    def predict(self, samples, callback, *, phase="planning"):
        total.require(self.model_loaded, "model must be loaded before prediction")
        return self.total.dispatch(phase, "predict", samples, callback)

    def backward(self, rows, callback, *, phase="planning"):
        total.require(self.model_loaded, "model must be loaded before backward")
        return self.total.dispatch(phase, "backward", rows, callback)

    def algorithm(self, kind, units, callback, *, phase="planning"):
        total.require(kind in ALGORITHM_KINDS, "unmapped planner operation")
        return self.total.dispatch(phase, ALGORITHM_KINDS[kind], units, callback)

    def physical_call(self, resource, callback):
        return self.physical.dispatch(resource, callback)

    def close(self):
        total.require(not self.closed, "host already closed")
        if not self.model_loaded:
            self.total.failed = True
        physical_receipt = self.physical.close()
        total_receipt = self.total.close()
        physical_audit = physical.verify_journal(
            physical_receipt["path"], physical_receipt["sha256"],
            total.digest(self.physical.contract))
        total_audit = total.verify_journal(
            total_receipt["journal"], total_receipt["sha256"],
            total.digest(self.total.contract))
        self.closed = True
        charged = total_audit["charged"]
        unused = {key: self.protocol["reservation"][key] - charged[key] for key in total.AXES}
        observed = {
            key for key, cap in self.protocol["reservation"].items()
            if cap > 0 and charged[key] > 0
        }
        required = {
            key for key, cap in self.protocol["reservation"].items()
            if cap > 0 and key not in self.protocol["conditional_axes"]
        }
        coverage_complete = required <= observed
        status = ("ready_receipt_unverified_cost" if total_audit["status"] == "closed_unverified"
                  and physical_audit["status"] == "closed_unverified"
                  and coverage_complete else "failed_or_incomplete")
        return {
            "schema": NAME + ".receipt.v1",
            "protocol_sha256": self.protocol_sha,
            "status": status,
            "reservation": deepcopy(self.protocol["reservation"]),
            "charged": charged,
            "unused": unused,
            "physical_charged": physical_audit["charged"],
            "coverage_complete": coverage_complete,
            "coverage_verified": False,
            "full_cost_certified": False,
            "fairness_proven": False,
            "independent_confirmation": False,
            "unverified_bypasses": list(UNVERIFIED_BYPASSES),
            "total_journal": total_receipt,
            "physical_journal": physical_receipt,
        }


def run_host(directory, protocol, expected_sha, planner):
    """Run a planner callback against the only supported host API."""
    total.require(callable(planner), "planner callback")
    host = Host(directory, protocol, expected_sha)
    result = None
    error = None
    try:
        result = planner(host)
    except BaseException as exc:
        host.total.failed = True
        host.physical.failed = True
        error = exc
    receipt = host.close()
    if error is not None:
        error.add_note("P1 total-budget receipt: " + receipt["status"])
        raise error
    return {"result": result, "receipt": receipt}
