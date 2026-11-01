"""Bridge a selected raw10 plan to the existing moving-root physical branch.

The caller owns authenticated task data, durable store, budget, construction,
native hooks, lifecycle and six-slot roster. This is not a production launcher.
"""
import numpy as np
from experiments.spectral_reachability import codex_p3_six_physical_contract_20261030_v1 as contract
from experiments.spectral_reachability import p1_constructive_moving_history_physical_20261027_v1 as moving


def run_branch(env, reference, source_plan, selection, row, ops, store, index):
    raw = contract.raw(selection["commands"])
    old_step = env.step
    had_own = "step" in env.__dict__
    acknowledgements = []

    def wrapped(command):
        result = old_step(command)
        scope = dict(ops.where)
        sent = np.asarray(command)
        actual = np.asarray(env.unwrapped.latest_action)
        # A return has happened even if the following validation fails.
        record = dict(ordinal=len(acknowledgements), sent_dtype=str(sent.dtype),
                      actual_dtype=str(actual.dtype), sent_shape=list(sent.shape),
                      actual_shape=list(actual.shape),
                      sent_hex=sent.tobytes().hex(), actual_hex=actual.tobytes().hex(),
                      terminated=bool(result[2]), truncated=bool(result[3]))
        acknowledgements.append(record)
        store.emit("step_ack_observed", **scope, receipt=record)
        if sent.dtype != np.float32 or actual.dtype != np.float32 or sent.shape != (2,) or actual.shape != (2,) or sent.tobytes() != actual.tobytes():
            raise ValueError("actual native raw ACK bytes")
        return result

    env.step = wrapped
    try:
        moving.branch(env, reference, source_plan, selection, row, ops, store, index)
    finally:
        if had_own:
            env.step = old_step
        else:
            delattr(env, "step")
    # Original branch also validates warm10/reset/goal/RGB and guards native stop.
    plan_acks = acknowledgements[10:]
    events = [dict(step=i, sent_float32_hex=a["sent_hex"], actual_float32_hex=a["actual_hex"],
                   terminated=a["terminated"], truncated=a["truncated"])
              for i, a in enumerate(plan_acks)]
    status = "horizon_completed" if row["status"] == "horizon_completed" else (
        "native_stopped" if row["status"] in ("native_success", "native_stopped") else "failed_partial")
    outcome = contract.verify_ack({"commands": raw.astype(float).tolist()}, events,
                                  {"status": status, "returned_steps": len(events)})
    store.emit("selected_ack_summary", **ops.where, events=events, outcome=outcome)
    return outcome
