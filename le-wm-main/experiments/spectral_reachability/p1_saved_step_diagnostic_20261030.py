"""Descriptive first-order audit of a sealed P1 suffix optimization trajectory.

This does not recompute model gradients or authenticate upstream data. Run the
original saved consumer separately before interpreting its output.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED_SHA = "aa162a9ff2b8fb66d13fbc8d3db625c753b9f090160d6101e38b1e300e291799"


def flatten(value):
    if isinstance(value, list):
        return [v for part in value for v in flatten(part)]
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite value")
    return [value]


def analyze(report):
    rows = report["rows"]
    if len(rows) != 160:
        raise ValueError("expected 160 candidates")
    increments = descending = uphill_descent = affected = 0
    largest = None
    steps = 0
    for row in rows:
        evaluations = row["evaluations"]
        if len(evaluations) != 9:
            raise ValueError("expected nine iterates")
        row_increases = 0
        for i, (a, b) in enumerate(zip(evaluations, evaluations[1:])):
            if a["iteration"] != i or b["iteration"] != i + 1:
                raise ValueError("iterate order")
            x, y, raw, grad = map(flatten, (a["logits"], b["logits"], a["raw"], a["raw_gradient"]))
            if any(len(v) != 20 for v in (x, y, raw, grad)):
                raise ValueError("expected two 10-dimensional packets")
            before, after = float(a["regularized"]), float(b["regularized"])
            if not math.isfinite(before) or not math.isfinite(after):
                raise ValueError("nonfinite objective")
            # This is the producer's saved update direction, not an independently
            # recomputed gradient through the frozen model or float32 tanh.
            dot = math.fsum((g * (1 - t * t) + 2 * .002 * z / 192) * (v - z)
                            for z, v, t, g in zip(x, y, raw, grad))
            delta = after - before
            steps += 1
            increments += delta > 0
            descending += dot < 0
            uphill_descent += delta > 0 and dot < 0
            row_increases += delta > 0
            if largest is None or delta > largest["delta"]:
                largest = dict(delta=delta, direction_dot=dot, slot=row["slot"],
                               arm=row["arm"], start=row["start"],
                               prefix_iteration=row["iteration"], suffix_iteration=i)
        affected += row_increases > 0
    return dict(candidates=len(rows), steps=steps, objective_increases=increments,
                first_order_descent=descending, increases_despite_descent=uphill_descent,
                candidates_with_increase=affected, largest_increase=largest,
                gradient_recomputed=False, physical_evidence=False, P1_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    data = args.report.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA:
        raise ValueError("sealed report hash mismatch")
    print(json.dumps(analyze(json.loads(data)), sort_keys=True))


if __name__ == "__main__":
    main()
