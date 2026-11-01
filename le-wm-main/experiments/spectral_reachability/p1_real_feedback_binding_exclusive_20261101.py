"""Fail-closed binding primitive for a future physical carry integration.

This module does not execute a planner or an environment.  Its inputs must be
obtained from the physical driver's own returned-action and observation records.
"""
import numpy as np

from . import p1_carry_binding_20261101 as binding


def bind_returned_packet(plan, *, episode, boundary, returned_actions,
                         observation):
    """Bind the preceding full plan to returned raw actions and current RGB.

    Event ordering is checked within the physical driver's journal; the caller
    must also verify that the supplied records actually come from that journal.
    """
    if not isinstance(plan, binding.BoundPlan):
        raise ValueError("sealed previous plan required")
    if not isinstance(observation, dict) or observation.get("boundary") != boundary:
        raise ValueError("observation boundary mismatch")
    observed = observation.get("observed_index")
    encoded = observation.get("encoded_index")
    if (type(observed) is not int or type(encoded) is not int
            or observed < 0 or encoded <= observed):
        raise ValueError("observation and encoding order required")
    if not isinstance(returned_actions, list) or len(returned_actions) != 5:
        raise ValueError("five returned actions required")
    values, indices = [], []
    for offset, record in enumerate(returned_actions, 1):
        if (not isinstance(record, dict) or record.get("episode_id") != episode
                or record.get("step") != plan.boundary + offset
                or type(record.get("index")) is not int):
            raise ValueError("returned action identity mismatch")
        values.append(record.get("command"))
        indices.append(record["index"])
    # The driver captures the last RGB before emitting the fifth control_return.
    if (not all(a < b for a, b in zip(indices, indices[1:]))
            or indices[-2] >= observed or indices[-1] >= encoded):
        raise ValueError("returned actions and observation must precede encoding")
    packet = np.asarray(values, dtype=np.float32)
    return plan.bind_feedback(episode=episode, boundary=boundary,
                              issued_packet=packet,
                              rgb_sha256=observation.get("pixel_sha256"))
