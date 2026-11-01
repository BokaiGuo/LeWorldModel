"""Six-slot raw10 contract and saved ACK verifier; no environment dispatch."""
import hashlib
import math
import numpy as np

from experiments.spectral_reachability import codex_p3_noshadow_physical_bind_20261030_v1 as binding

NAME = "codex_p3_six_physical_contract_20261030_v1"
ORDER = [(root, policy) for root in (0, 1) for policy in binding.POLICIES]


def contract():
    return {"schema": NAME + ".contract", "order": [list(x) for x in ORDER],
            "requested_steps": 10, "warmup_steps": 10, "physical_replans": 0,
            "action": "exact saved H2 winner raw float32; no scaling, clipping, padding or substitution",
            "stop": "native terminated or truncated: stop immediately and retain partial; no retry",
            "execution": "fresh environment per slot, original reset placement and goal plus warmup source guard",
            "caps": {"environment_construct": 6, "environment_close": 6, "reset": 12,
                     "public_step": 120, "physics_step": 1224, "set_state": 24,
                     "goal_render": 12, "goal_bind": 6, "explicit_render": 132},
            "physics_step_scope": "hooked space.step, not a CUDA or FLOP measure",
            "constructor_internal_work": "unknown if constructor fails before returning",
            "physical_authorized": False, "scientific_pass": False}


def raw(value):
    return binding.host.prior.prior.raw_commands(value)


def validate_plan(rows):
    if type(rows) is not list or [(r["root"], r["policy"]) for r in rows] != ORDER:
        raise ValueError("six-slot fixed order")
    for row in rows:
        action = raw(row["commands"])
        if hashlib.sha256(action.tobytes()).hexdigest() != row["raw_float32_sha256"]:
            raise ValueError("raw action bits")
        if row.get("physical_authorized") is not False:
            raise ValueError("producer does not authorize physical")
    return True


def verify_ack(plan, events, outcome):
    """Validate a saved ordered actual action prefix, no imputed endpoint.

    Inputs require independent authentication by the future host runner. This
    function alone cannot establish that events came from a real environment.
    """
    expected = raw(plan["commands"])
    if type(events) is not list or type(outcome) is not dict:
        raise ValueError("saved records")
    stopped = False
    for step, event in enumerate(events):
        if stopped or step >= 10 or type(event) is not dict or set(event) != {
            "step", "sent_float32_hex", "actual_float32_hex", "terminated", "truncated"
        } or type(event["step"]) is not int or event["step"] != step:
            raise ValueError("ACK sequence")
        if type(event["terminated"]) is not bool or type(event["truncated"]) is not bool:
            raise ValueError("native stop flags")
        bits = expected[step].tobytes().hex()
        if event["sent_float32_hex"] != bits or event["actual_float32_hex"] != bits:
            raise ValueError("ACK raw bit mismatch")
        stopped = event["terminated"] or event["truncated"]
    if type(outcome.get("returned_steps")) is not int or outcome["returned_steps"] != len(events):
        raise ValueError("returned step census")
    status = outcome.get("status")
    if status == "horizon_completed":
        if len(events) != 10 or stopped:
            raise ValueError("full horizon")
    elif status == "native_stopped":
        if not stopped:
            raise ValueError("native stop")
    elif status == "failed_partial":
        pass
    else:
        raise ValueError("unknown terminal")
    return {"status": status, "returned_steps": len(events), "requested_steps": 10,
            "missing_steps": 10-len(events), "padded": False,
            "physical_certified": False, "scientific_pass": False}
