"""Opt-in bridge primitives; refuse live host execution with known gaps."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from . import p1_prospective_total_guard_20261101_leibniz as total
from . import codex_four_arm_durable_20261030 as legacy

NAME = "p1_total_guard_host_bridge_20261101_leibniz"
NODE_KINDS = {
    "encode": ("source_encode", "encode_calls", "encode_samples"),
    "action_encode": ("action_encode", "action_encode_calls", None),
    "predict": ("predict", "predict_calls", "predict_samples"),
    "backward": ("backward", "backward_calls", "vjp_rows"),
    "jacobian": ("jacobian", "jacobian_calls", None),
    "svd": ("svd", "svd_calls", None),
    "solve": ("solve", "solve_calls", None),
    "selection": ("candidate", "selection_evaluations", None),
    "replay": ("replay", "replay_calls", None),
    "physical": ("physics_step", "physical_steps", None),
}
PHYSICAL_KINDS = {
    "reset": "reset", "public_step": "physical_dispatch", "physics_step": "physics_step",
    "explicit_render": "render", "goal_render": "render",
}
MISSING = (
    "visual_sink_direct_begin_end_not_guarded",
    "constructor_close_goal_bind_goal_setter_set_state_missing_total_axes",
    "offline_goal_generation_and_encoder_upstream_not_reexecuted",
    "autograd_functional_and_physics_bypass_coverage_not_independently_verified",
    "common_full_run_caps_and_time_memory_policy_not_precommitted",
)


def gate():
    return {"schema": NAME + ".coverage.v1", "host_execution_authorized": False,
            "coverage_verified": False, "full_cost_certified": False, "fairness_proven": False,
            "missing": list(MISSING)}


def require_host_ready():
    raise ValueError("fail-closed: " + ", ".join(MISSING))


def node_dispatch(guard, original, driver, phase, kind, reservation, fn):
    """Route one actual legacy driver callback through the prospective guard."""
    if kind not in NODE_KINDS or not callable(fn):
        guard.fail("unmapped node operation")
    legacy.spec.vector(reservation)
    target, primary, sample = NODE_KINDS[kind]
    allowed = {primary} | ({sample} if sample else set())
    if not all(reservation[k] == 0 for k in legacy.spec.AXES if k not in allowed):
        guard.fail("mixed legacy reservation cannot be mapped")
    count = reservation[primary]
    total.require(type(count) is int and count > 0, "positive node charge")
    if sample:
        total.require(count == 1 and type(reservation[sample]) is int and reservation[sample] > 0,
                      "actual samples/directions")
        units = reservation[sample]
    else:
        units = count
    phase_map = {"setup": "arm_setup", "initialization": "arm_setup", "jacobian": "planning",
                 "screening": "planning", "trial": "planning", "common_suffix": "planning",
                 "verification": "verification", "physical": "physical"}
    total.require(phase in phase_map, "legacy phase")
    return guard.dispatch(phase_map[phase], target, units,
                          lambda: original(driver, phase, kind, reservation, fn))


def physical_call(guard, original, meter, resource, fn):
    """Route callback-style meter entries; reject unsupported setup operations."""
    if resource not in PHYSICAL_KINDS or not callable(fn):
        guard.fail("unmapped physical resource: " + str(resource))
    return guard.dispatch("physical", PHYSICAL_KINDS[resource], 1, lambda: original(meter, resource, fn))


def guarded_load(guard, loader, *args, **kwargs):
    """Call only actual loader; install sentinels before releasing model."""
    model = guard.dispatch("shared_setup", "model_load", 1, lambda: loader(*args, **kwargs))
    guard.install_model_sentinels(model)
    return model


def guarded_visual(guard, handler, request, sink, **runtime):
    """Wrap whole native RGB encoding; actual module sentinels check callback."""
    return guard.dispatch("feedback", "source_encode", 1,
                          lambda: handler(request, sink, **runtime))


@contextmanager
def node_patch(guard):
    """Scoped process-local patch of the exact class used by both legacy planners."""
    original = legacy.Driver.dispatch

    def guarded(driver, phase, kind, reservation, fn):
        return node_dispatch(guard, original, driver, phase, kind, reservation, fn)

    with patch.object(legacy.Driver, "dispatch", guarded):
        yield


def pair_protocol(structured, cem):
    """Validate draft pair only; no precommitment/coverage assertion."""
    pair = total.validate_pair(structured, cem)
    total.check_sources(structured)
    total.check_sources(cem)
    total.require(str(Path(__file__).resolve()) in structured["source_sha256"]
                  and str(Path(__file__).resolve()) in cem["source_sha256"], "bridge source unbound")
    return {"schema": NAME + ".pair.v1", "structured": deepcopy(structured), "cem": deepcopy(cem),
            "structured_sha256": total.digest(structured), "cem_sha256": total.digest(cem),
            "pair_check": pair, "coverage": gate(), "P1_admission": False}
