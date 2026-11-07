"""Versioned clock fix for the historical timed-prefix planner.

The historical implementation is retained intact for artifact verification.
This adapter removes unscored trailing actions at every shortened horizon;
it does not change candidate generation, scoring, or selection.
"""
import numpy as np

from experiments.pc_rgb_timed_prefix_search_20261107_codex import compile_timed


def compile_timed_clock_fixed(boundary, current, *, scaler, old_tail, candidate_seed,
                              strategy_seed, strategy, remaining_steps):
    result = compile_timed(boundary, current, scaler=scaler, old_tail=old_tail,
                           candidate_seed=candidate_seed, strategy_seed=strategy_seed,
                           strategy=strategy, remaining_steps=remaining_steps)
    actions = np.asarray(result["actions"])
    expected = (remaining_steps, 2)
    if actions.shape != (25, 2):
        raise ValueError("unexpected historical candidate shape")
    actions = actions[:remaining_steps]
    if actions.shape != expected or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
        raise ValueError("selected actions do not match remaining physical clock")
    return {**result, "actions": actions.copy()}
