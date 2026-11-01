"""Fail-closed matched-cost sidecar, not a solver or scientific admission gate.

Standard-library only. Counters are caller-supplied measurements/reservations,
not instrumentation or CUDA attestation. Never infer FLOPs or physical efficacy.
All planning work (screening, rejected candidates, compilation and cache builds)
must be charged; evaluator work belongs in a separate ledger.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json


BINDINGS = (
    "root", "checkpoint", "source", "observation_access", "goal_metric",
    "action_domain", "temporal_execution", "evaluation", "calibration_split",
    "outcome_split", "cache_policy", "hardware_concurrency", "measurement_scope",
)
RESOURCES = (
    "predict_calls", "predict_samples", "predict_token_work", "encoder_samples",
    "scalar_backward_calls", "jacobian_vjp_rows", "jvp_calls",
    "candidate_cost_samples", "search_expansions", "planning_wall_ns",
)
LABELS = {"proxy", "local_algorithm", "faithfulness_audited"}


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: integer >= {minimum} required")
    return value


def _hash(value):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("lowercase SHA256 required; unknown is not a binding")
    return value


def _vector(value):
    if type(value) is not dict or set(value) != set(RESOURCES):
        raise ValueError("complete resource vector required; missing is not zero")
    return {k: _integer(value[k], k) for k in RESOURCES}


def protocol(*, bindings, feedback_steps, resource_caps, primary_resources):
    """Build JSON sidecar; hashes refer to externally frozen canonical artifacts.

    action_domain must include parameter AND effective physical bounds, mapping,
    dtype and initialization/proposal law. temporal_execution includes horizons,
    units, chunk/repeat, total steps and stopping. No implicit unit conversions.
    Equal primary caps mean equal ceilings, NOT equal actual work or FLOPs.
    """
    if type(bindings) is not dict or set(bindings) != set(BINDINGS):
        raise ValueError("all exact protocol bindings required")
    bindings = {k: _hash(bindings[k]) for k in BINDINGS}
    if bindings["calibration_split"] == bindings["outcome_split"]:
        raise ValueError("calibration and outcome manifests must differ")
    if type(feedback_steps) is not list:
        raise ValueError("explicit feedback steps required, including [] for open loop")
    steps = [_integer(v, "feedback step", 1) for v in feedback_steps]
    if steps != sorted(set(steps)):
        raise ValueError("feedback steps must be unique and increasing")
    if (type(primary_resources) is not list or not primary_resources
            or any(type(k) is not str or k not in RESOURCES for k in primary_resources)
            or len(set(primary_resources)) != len(primary_resources)):
        raise ValueError("explicit unique primary resources required")
    caps = _vector(resource_caps)
    if any(caps[k] == 0 for k in primary_resources):
        raise ValueError("primary resource caps must be positive")
    return {"schema": "d2p_baseline_protocol.v2", "bindings": bindings,
            "feedback_steps": steps, "resource_caps": caps,
            "primary_resources": sorted(primary_resources)}


def _validated(value):
    if type(value) is not dict or set(value) != {
        "schema", "bindings", "feedback_steps", "resource_caps", "primary_resources"
    } or value["schema"] != "d2p_baseline_protocol.v2":
        raise ValueError("invalid v2 protocol")
    return protocol(**{k: value[k] for k in value if k != "schema"})


def fingerprint(value):
    value = _validated(value)
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compare_protocols(left, right):
    """Report differences; never grant publication faithfulness or paired inference."""
    left, right = _validated(left), _validated(right)
    differences = [f"bindings.{k}" for k in BINDINGS
                   if left["bindings"][k] != right["bindings"][k]]
    for k in ("feedback_steps", "primary_resources"):
        if left[k] != right[k]:
            differences.append(k)
    for k in set(left["primary_resources"]) | set(right["primary_resources"]):
        if left["resource_caps"][k] != right["resource_caps"][k]:
            differences.append(f"resource_caps.{k}")
    return {"matched_protocol_and_primary_caps": not differences,
            "differences": sorted(differences),
            "nonprimary_caps_equal": all(
                left["resource_caps"][k] == right["resource_caps"][k]
                for k in RESOURCES
                if k not in set(left["primary_resources"]) | set(right["primary_resources"])),
            "equal_actual_compute": False, "scientific_stage_admission": False}


def method_identity(*, name, label, implementation_sha256, audit_sha256=None):
    """Record a claim, not verify it. Proxy labels cannot become official results.

    faithfulness_audited requires an external audit artifact mapping the pinned
    paper/official revision, mechanisms, hyperparameters and deviations. This
    function checks artifact identity syntax only, never its existence/content.
    """
    if type(name) is not str or not name.strip() or label not in LABELS:
        raise ValueError("explicit name and proxy/local_algorithm/faithfulness_audited label required")
    if label == "faithfulness_audited":
        _hash(audit_sha256)
    elif audit_sha256 is not None:
        raise ValueError("audit claim requires faithfulness_audited label")
    return {"name": name, "label": label,
            "implementation_sha256": _hash(implementation_sha256),
            "audit_sha256": audit_sha256, "official_reproduction_verified_here": False}


class BudgetLedger:
    """Serial outer-runner reservation ledger; not safe for concurrent callbacks.

    Reserve upper bounds BEFORE work; complete with measured full vector AFTER.
    Failed attempts keep their reservation charged and poison final admission.
    A callback exceeding its reservation poisons admission even below total cap.
    Neural execution is permitted only with explicit host_cuda metadata, which
    the caller must independently attest; this class launches no computation.
    """

    def __init__(self, frozen_protocol, identity, *, execution_backend):
        if execution_backend != "host_cuda":
            raise ValueError("host CUDA only; no CPU fallback")
        self._protocol = _validated(frozen_protocol)
        if type(identity) is not dict or set(identity) != {
            "name", "label", "implementation_sha256", "audit_sha256",
            "official_reproduction_verified_here"
        } or identity["official_reproduction_verified_here"] is not False:
            raise ValueError("invalid method identity")
        self._identity = method_identity(**{k: identity[k] for k in identity
                                           if k != "official_reproduction_verified_here"})
        self._events = []
        self._feedback = []
        self._failed = False
        self._closed = False

    def _open(self):
        if self._failed or self._closed:
            raise RuntimeError("ledger failed or closed")

    def _charged(self):
        return {k: sum(e.get("actual", e["reserved"])[k] for e in self._events)
                for k in RESOURCES}

    def reserve(self, purpose, upper_bound):
        self._open()
        if self._events and self._events[-1]["status"] == "pending":
            raise RuntimeError("complete pending work first")
        if type(purpose) is not str or not purpose.strip():
            raise ValueError("work purpose required")
        upper_bound = _vector(upper_bound)
        charged = self._charged()
        if any(charged[k] + upper_bound[k] > self._protocol["resource_caps"][k]
               for k in RESOURCES):
            self._failed = True
            raise ValueError("budget exceeded before dispatch")
        self._events.append({"purpose": purpose, "reserved": upper_bound, "status": "pending"})
        return len(self._events) - 1

    def complete(self, ticket, actual=None, *, succeeded=True):
        self._open()
        _integer(ticket, "ticket")
        if (ticket != len(self._events) - 1 or not self._events
                or self._events[ticket]["status"] != "pending"):
            raise ValueError("current pending ticket required")
        event = self._events[ticket]
        if succeeded is not True:
            event["status"] = "failed"
            self._failed = True
            return
        try:
            measured = _vector(actual)
        except ValueError:
            self._failed = True
            event["status"] = "invalid_measurement"
            raise
        event["actual"] = measured
        if any(measured[k] > event["reserved"][k] for k in RESOURCES):
            self._failed = True
            event["status"] = "reservation_overrun"
            raise ValueError("actual work exceeded reservation")
        event["status"] = "completed"

    def feedback(self, primitive_step):
        self._open()
        _integer(primitive_step, "primitive step", 1)
        index = len(self._feedback)
        expected = self._protocol["feedback_steps"]
        if (any(e["status"] == "pending" for e in self._events)
                or index >= len(expected) or expected[index] != primitive_step):
            self._failed = True
            raise ValueError("feedback schedule mismatch or pending work")
        self._feedback.append(primitive_step)

    def receipt(self, *, finalize=False):
        if type(finalize) is not bool:
            raise ValueError("finalize must be boolean")
        if finalize:
            self._closed = True
        complete = (self._closed and not self._failed and bool(self._events)
                    and all(e["status"] == "completed" for e in self._events)
                    and self._feedback == self._protocol["feedback_steps"])
        return deepcopy({"schema": "d2p_baseline_budget_receipt.v2",
            "protocol_sha256": fingerprint(self._protocol), "method": self._identity,
            "events": self._events, "charged_resources": self._charged(),
            "feedback_steps": self._feedback, "completed": complete,
            "failed": self._failed, "closed": self._closed,
            "measurement_origin": "caller_supplied_not_independently_attested",
            "physical_efficacy_verified": False, "scientific_stage_admission": False})
