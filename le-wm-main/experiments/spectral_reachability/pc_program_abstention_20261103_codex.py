"""Explicit no-reconfiguration decision for complete-program selection.

The fallback is the optimized zero-prefix program, not a guarantee of a
physical improvement or a no-action policy.
"""
import numpy as np

from experiments.spectral_reachability.pc_goal_conditioned_program_20261103_codex import (
    compile_program,
)


def select_with_abstention(programs, minimum_gain):
    """Select a nonzero-prefix program only if it improves on zero by the threshold."""
    if not np.isfinite(minimum_gain) or minimum_gain < 0 or not programs:
        raise ValueError("invalid abstention policy")
    names = [item["name"] for item in programs]
    if len(set(names)) != len(names) or names.count("zero") != 1:
        raise ValueError("expected unique names and exactly one zero control")
    for item in programs:
        prefix = np.asarray(item["prefix"], dtype=float)
        suffix = np.asarray(item["suffix"], dtype=float)
        if (prefix.shape != (5, 2) or suffix.shape != (10, 2)
                or not np.isfinite(prefix).all() or not np.isfinite(suffix).all()
                or np.any(np.abs(prefix) > 1) or np.any(np.abs(suffix) > 1)
                or not np.isfinite(item["terminal_score"])):
            raise ValueError("invalid full program")
    zero = next(item for item in programs if item["name"] == "zero")
    if np.any(np.asarray(zero["prefix"]) != 0):
        raise ValueError("zero control is not a zero prefix")
    best = min(programs, key=lambda item: item["terminal_score"])
    gain = float(zero["terminal_score"] - best["terminal_score"])
    abstained = best["name"] == "zero" or gain <= minimum_gain
    return {"selected": zero if abstained else best, "abstained": abstained,
            "minimum_gain": float(minimum_gain), "best_gain_vs_zero": gain,
            "fallback": "optimized_zero_prefix"}


def compile_with_abstention(*args, minimum_gain, **kwargs):
    """Compile once, then bind the executed selection to the fallback policy."""
    result = compile_program(*args, **kwargs)
    decision = select_with_abstention(result["programs"], minimum_gain)
    result["selected"] = decision["selected"]
    # The upstream flag was calculated for the pre-abstention selection.
    # It must never be reused as a claim about the final executed program.
    result.pop("qualified_against_zero_only", None)
    result["abstention"] = {key: value for key, value in decision.items() if key != "selected"}
    return result
