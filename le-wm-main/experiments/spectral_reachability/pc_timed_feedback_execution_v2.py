"""Fixed-terminal-horizon feedback using the common v17/CEM execution path."""
from dataclasses import replace

import numpy as np

from experiments.spectral_reachability import pc_timed_program_execution_v2 as pc_timed_program_execution


def run(initial_context, config, *, arm, execution_steps, prepare, seal_decision,
        dispatch, baseline_seed=0, synchronize=None):
    """Execute packets with fresh context and a decision sealed before dispatch.

    ``prepare`` receives only the current planner context and returns a
    predictor and target. ``seal_decision`` must durably persist the complete
    decision before ``dispatch`` is invoked. A failure propagates immediately;
    partial physical execution must never be retried implicitly.
    """
    horizon = config.segment_length * config.segments
    packet = config.model_packet_length
    if (execution_steps <= 0 or execution_steps % packet or horizon % execution_steps
            or execution_steps != config.segment_length):
        raise ValueError("unsupported fixed-endpoint execution clock")
    context, carry, decisions = initial_context, None, []
    for step in range(horizon // execution_steps):
        remaining = horizon - step * execution_steps
        current = replace(config, segments=remaining // config.segment_length,
                          seed=config.seed + step * 10000)
        predictor, target, binding = prepare(context, step)
        plan = pc_timed_program_execution.execute(
            predictor, target, current, arm=arm, carry=carry,
            baseline_seed=baseline_seed + step, synchronize=synchronize)
        selected = np.asarray(plan["result"]["selected"], dtype=np.float32)
        if (selected.shape != (remaining, config.action_dim) or not np.isfinite(selected).all()
                or np.any(np.abs(selected) > 1)):
            raise ValueError("planner returned invalid chain")
        bank = plan["traces"][0]
        carry_score = None
        if carry is not None:
            # proposal_bank puts the carry at index 1 and re-evaluates it on
            # the current observation for both arms.
            if not np.array_equal(bank["actions"][1], carry):
                raise ValueError("carry was not rescored in current context")
            carry_score = float(bank["scores"][1])
        decision = {"step": step, "remaining_steps": remaining, "context_binding": binding,
                    "carry_in": None if carry is None else carry.tolist(), "carry_score": carry_score,
                    "selected": selected.tolist(), "dispatch": selected[:execution_steps].tolist(),
                    "resource": plan["resource"], "calls": plan["calls"],
                    "planner_result": plan["result"], "traces": plan["traces"]}
        receipt = seal_decision(decision)
        if receipt is None:
            raise ValueError("decision was not sealed")
        context, event = dispatch(context, selected[:execution_steps].copy(), receipt)
        carry = selected[execution_steps:].copy() if remaining > execution_steps else None
        decisions.append({"step": step, "receipt": receipt, "event": event})
    return {"events": decisions, "final_context": context}
