"""Common metered execution for the v17 compiler and its shared-bank control.

This is a development runner, not a scientific gate.  The predictor is the
only source of planner features and scores; physical evaluation is external.
"""
from time import perf_counter

import numpy as np

from experiments.spectral_reachability import pc_global_multistart_cem
from experiments.spectral_reachability import pc_global_reachability_v18
from experiments.spectral_reachability import pc_recursive_response_v17


def execute(predict, target, config, *, arm, score_scale=1.0, score_target=0.0,
            carry=None, baseline_seed=0, synchronize=None, clock=perf_counter):
    """Plan once and return the result, actual call ledger, and scored actions.

    ``synchronize`` must synchronize the relevant device before reading the
    clock.  With no callback timings describe synchronous CPU execution only.
    Trace collection is included in planning time; persistence is not.
    """
    if arm not in ("v17", "v18", "multistart_cem"):
        raise ValueError("unsupported planner")
    if config.model_packet_length <= 0 or config.token_cap <= 0:
        raise ValueError("invalid packet length or budget")
    sync = synchronize if synchronize is not None else lambda: None
    calls, traces = [], []
    used = 0

    def metered(actions):
        nonlocal used
        raw = np.asarray(actions, dtype=np.float32)
        if (raw.ndim != 3 or len(raw) == 0 or raw.shape[2] != config.action_dim
                or raw.shape[1] == 0 or raw.shape[1] % config.model_packet_length
                or not np.isfinite(raw).all() or np.any(np.abs(raw) > 1)):
            raise ValueError("invalid action batch or packet boundary")
        cost = len(raw) * (raw.shape[1] // config.model_packet_length)
        if used + cost > config.token_cap:
            raise RuntimeError("prediction token budget exceeded")
        used += cost
        sync()
        before = clock()
        try:
            features, scores = predict(raw.copy())
            sync()
            duration = clock() - before
        except BaseException:
            sync()
            calls.append({"samples": len(raw), "physical_steps": raw.shape[1],
                          "prediction_tokens": cost, "seconds": clock() - before,
                          "status": "failed"})
            raise
        features, scores = np.asarray(features), np.asarray(scores)
        calls.append({"samples": len(raw), "physical_steps": raw.shape[1],
                      "prediction_tokens": cost, "seconds": duration, "status": "completed"})
        traces.append({"actions": raw.copy(), "features": features.copy(), "scores": scores.copy()})
        return features, scores

    sync()
    started = clock()
    if arm in ("v17", "v18"):
        compiler = pc_recursive_response_v17 if arm == "v17" else pc_global_reachability_v18
        result = compiler.compile_program(
            metered, target, config, score_scale=score_scale, score_target=score_target, carry=carry)
        reported_tokens = result["resource"]["prediction_tokens"]
    else:
        result = pc_global_multistart_cem.plan(metered, config, seed=baseline_seed, carry=carry)
        reported_tokens = result["prediction_tokens"]
    sync()
    planning_seconds = clock() - started
    if reported_tokens != used:
        raise RuntimeError("planner and actual prediction token counts differ")
    return {"result": result, "calls": calls, "traces": traces,
            "resource": {"prediction_tokens": used, "prediction_calls": len(calls),
                         "prediction_seconds": sum(call["seconds"] for call in calls),
                         "planning_seconds_with_trace_capture": planning_seconds,
                         "device_synchronized": synchronize is not None,
                         "serialization_included": False}}
