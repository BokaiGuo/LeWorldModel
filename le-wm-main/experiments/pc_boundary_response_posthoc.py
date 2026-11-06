"""Feasible finite differences at the actual float32 nominal action.

This module is a posthoc measurement tool, not a planner or gate decision.
"""
import numpy as np


def probes(nominal, epsilon, dimensions=10):
    """Return nominal and feasible perturbations with actual denominators."""
    x = np.asarray(nominal, dtype=np.float32)
    if not np.all(np.isfinite(x)) or np.any(np.abs(x) > 1):
        raise ValueError("nominal must be finite and in [-1,1]")
    if not np.isfinite(epsilon) or epsilon <= 0 or dimensions > x.size:
        raise ValueError("invalid probe specification")
    actions = [x.copy()]
    specs = []
    for i in range(dimensions):
        v = float(x.flat[i])
        choices = []
        for direction in (1, -1):
            target = v + direction * epsilon
            if -1 <= target <= 1:
                y = x.copy()
                y.flat[i] = np.float32(target)
                delta = float(y.flat[i]) - v
                if delta * direction > 0 and abs(float(y.flat[i])) <= 1:
                    choices.append((direction, len(actions), delta))
                    actions.append(y)
        if not choices:
            raise ValueError(f"no representable full-step probe at coordinate {i}")
        specs.append({"coordinate": i, "kind": "central_secant" if len(choices) == 2 else
                      ("forward" if choices[0][0] == 1 else "backward"),
                      "samples": [{"direction": d, "index": j, "delta": h} for d, j, h in choices]})
    return np.stack(actions), specs


def differences(values, branches, specs):
    """Compute secants and report branch crossings; never imply smoothness."""
    values, branches = np.asarray(values), np.asarray(branches)
    columns, stability = [], []
    for spec in specs:
        samples = spec["samples"]
        if len(samples) == 2:
            a, b = samples
            column = (values[a["index"]] - values[b["index"]]) / (a["delta"] - b["delta"])
        else:
            a = samples[0]
            column = (values[a["index"]] - values[0]) / a["delta"]
        columns.append(column)
        stability.append(all(np.array_equal(branches[a["index"]], branches[0]) for a in samples))
    return np.stack(columns, axis=1), stability
