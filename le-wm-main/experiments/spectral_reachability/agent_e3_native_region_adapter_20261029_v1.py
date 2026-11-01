"""Scoped native meter region for the single-capture E3 packet.

This adapter is not a dispatcher or a saved semantic consumer.  The caller
must bind inputs, persist events, enforce budgets and close the environment.
"""
from experiments.spectral_reachability import agent_e3_feedback_single_capture_20261029_v1 as original


def packet(env, commands, *, start, row, goal, goal_rgb, meter, store, ops=original.old):
    original.need(type(start) is int and start in (0, 5, 10, 15), 'fixed control boundary')
    prior = meter.region
    meter.region = 'prefix' if start == 0 else 'suffix'
    try:
        return original.packet(env, commands, start=start, row=row, goal=goal,
                               goal_rgb=goal_rgb, meter=meter, store=store, ops=ops)
    finally:
        meter.region = prior
