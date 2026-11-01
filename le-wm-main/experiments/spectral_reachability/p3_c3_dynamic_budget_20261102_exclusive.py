"""CPU-only P3/C3 dynamic-budget protocol kernel.

The kernel fixes quota allocation before any outcome and uses exact integer
largest-remainder allocation.  It does not import a model, CUDA, or an
environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import argparse
import json
import os
from pathlib import Path


CALL_CLASSES = (
    "action_encoder",
    "predict",
    "predictor_sample",
    "observation_encode",
    "planner_overhead",
    "physical_action",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive_fraction(value):
    if isinstance(value, bool):
        raise ValueError("boolean weight")
    result = Fraction(value)
    require(result > 0, "positive weight required")
    return result


def allocate(total, weights, tie_order=None):
    """Allocate exactly ``total`` units without float rounding.

    Largest remainders are resolved by the declared stable order.  This makes
    allocation a pre-outcome function of the budget point and strategy only.
    """
    require(type(total) is int and total >= 0, "nonnegative integer total")
    require(type(weights) in (list, tuple) and weights, "nonempty weights")
    fractions = [_positive_fraction(value) for value in weights]
    order = list(range(len(fractions))) if tie_order is None else list(tie_order)
    require(sorted(order) == list(range(len(fractions))), "complete tie order")
    denominator = sum(fractions)
    exact = [Fraction(total) * value / denominator for value in fractions]
    result = [value.numerator // value.denominator for value in exact]
    remaining = total - sum(result)
    ranking = sorted(range(len(result)), key=lambda index: (
        -(exact[index] - result[index]), order.index(index)
    ))
    for index in ranking[:remaining]:
        result[index] += 1
    require(sum(result) == total, "allocation conservation")
    return result


@dataclass(frozen=True)
class BudgetPanel:
    points: tuple
    weights: tuple
    tie_order: tuple = (0, 1)
    fixed_allocations: tuple | None = None

    def __post_init__(self):
        require(self.points and all(type(point) is int and point > 0 for point in self.points),
                "positive budget points")
        require(len(self.weights) == 2 and len(self.tie_order) == 2,
                "two-root panel required")
        _positive_fraction(self.weights[0])
        _positive_fraction(self.weights[1])
        require(tuple(sorted(self.tie_order)) == (0, 1), "two-root tie order")
        if self.fixed_allocations is not None:
            require(len(self.fixed_allocations) == len(self.points),
                    "one fixed allocation per point")
            for point, allocation in zip(self.points, self.fixed_allocations):
                require(len(allocation) == 2 and sum(allocation) == point
                        and all(type(value) is int and value >= 0 for value in allocation),
                        "valid fixed allocation")

    def freeze(self):
        rows = []
        for index, point in enumerate(self.points):
            signal = (
                list(self.fixed_allocations[index])
                if self.fixed_allocations is not None
                else allocate(point, self.weights, self.tie_order)
            )
            reverse = list(reversed(signal))
            require(signal != reverse, "strategies collapsed after allocation")
            require(sum(signal) == sum(reverse) == point, "equal total budget")
            rows.append({
                "budget_point": point,
                "signal": signal,
                "reverse": reverse,
                "candidate_total_equal": True,
                "strategy_difference_precommitted": True,
            })
        return {
            "schema": "p3_c3_dynamic_budget_20261102_exclusive.protocol.v1",
            "points": rows,
            "weights": [str(Fraction(value)) for value in self.weights],
            "tie_order": list(self.tie_order),
            "selection_after_outcome": False,
            "P3_admission": False,
            "C3_admission": False,
            "gate_modified": False,
        }


class CallLedger:
    """Classified non-refundable call accounting for one frozen budget arm."""

    def __init__(self, caps):
        require(set(caps) == set(CALL_CLASSES), "complete call-class caps")
        require(all(type(value) is int and value >= 0 for value in caps.values()),
                "nonnegative call caps")
        self.caps = dict(caps)
        self.attempted = dict.fromkeys(CALL_CLASSES, 0)
        self.returned = dict.fromkeys(CALL_CLASSES, 0)
        self.events = []
        self.closed = False

    def dispatch(self, category, operation):
        require(category in CALL_CLASSES and callable(operation), "classified callable")
        require(not self.closed, "ledger closed")
        require(self.attempted[category] < self.caps[category], "call cap exceeded")
        self.attempted[category] += 1
        event = {"category": category, "attempt": self.attempted[category],
                 "returned": False}
        self.events.append(event)
        try:
            value = operation()
        except BaseException:
            self.closed = True
            raise
        self.returned[category] += 1
        event["returned"] = True
        return value

    def close(self):
        self.closed = True
        return {
            "attempted": dict(self.attempted),
            "returned": dict(self.returned),
            "complete": self.attempted == self.returned,
            "P3_admission": False,
            "C3_admission": False,
        }


def protocol():
    # Explicit effective quotas are frozen before outcomes.  They are not
    # recomputed from rounded floating-point shares.
    panel = BudgetPanel((20, 30), (Fraction(3, 1), Fraction(1, 1)),
                        fixed_allocations=((15, 5), (22, 8)))
    frozen = panel.freeze()
    frozen["call_classes"] = list(CALL_CLASSES)
    frozen["accounting"] = {
        "attempted_and_returned_separate": True,
        "failed_calls_are_not_refunded": True,
        "axes_are_not_summed_as_flops": True,
        "wall_time_and_energy": "unknown",
    }
    return frozen


def compare(point, signal, reverse):
    """Classify actual equality; never treat missing or failed calls as zero."""
    rows = {item["budget_point"]: item for item in protocol()["points"]}
    require(type(point) is int and point in rows, "frozen budget point")
    expected = rows[point]
    for arm, record in (("signal", signal), ("reverse", reverse)):
        require(type(record) is dict and record.get("allocation") == expected[arm],
                "frozen arm allocation")
        for field in ("attempted", "returned"):
            vector = record.get(field)
            require(type(vector) is dict and set(vector) == set(CALL_CLASSES)
                    and all(type(x) is int and x >= 0 for x in vector.values()),
                    "complete resource vector")
        require(all(record["returned"][k] <= record["attempted"][k]
                    for k in CALL_CLASSES), "returns exceed attempts")
    complete = all(row["attempted"] == row["returned"] for row in (signal, reverse))
    return {
        "budget_point": point,
        "equal_candidate_total": True,
        "equal_recorded_axes": complete and signal["attempted"] == reverse["attempted"],
        "both_complete": complete,
        "equal_total_compute": False,
        "flops": "unknown",
        "energy": "unknown",
        "P3_admission": False,
        "C3_admission": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).absolute()
    require(output.parent.name == "results" and output.name.startswith("p3_c3_")
            and not os.path.lexists(output), "exclusive output")
    payload = (json.dumps(protocol(), indent=2, sort_keys=True) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output": str(output), "bytes": len(payload),
                      "P3_admission": False, "C3_admission": False}))


if __name__ == "__main__":
    main()
