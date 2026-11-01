"""Prospective equal-panel quota intervention; callback seam, not a host launcher.

Every arm evaluates the same ordered 15+15 candidate identities. Quotas only
control eligibility, not computation. This is a new exploration, not the old
adaptive allocator, and no callback is authenticated by this module.
"""
import math

POLICIES = ("uniform", "signal", "reverse")
PANEL = tuple((group, index) for group in (0, 1) for index in range(15))
EXPECTED = {"candidate": 30, "predict": 60, "metric": 30}


def allocation(policy, weights):
    if policy not in POLICIES or type(weights) not in (list, tuple) or len(weights) != 2:
        raise ValueError("policy or weights")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in weights):
        raise ValueError("finite weights")
    if policy == "uniform" or weights[0] == weights[1]:
        return (10, 10)
    high = 0 if weights[0] > weights[1] else 1
    if policy == "reverse":
        high = 1 - high
    return (15, 5) if high == 0 else (5, 15)


def run(policy, weights, evaluate, event):
    """Evaluate all 30 H2 candidates independently, then apply fixed eligibility.

    evaluate(group,index) must return exactly {cost,predict_calls}; its accounting
    is self-reported and must be independently reconciled in a real backend.
    event is a caller-owned durable sink; exceptions propagate without retries.
    """
    quotas = allocation(policy, weights)
    if not callable(evaluate) or not callable(event):
        raise ValueError("callbacks required")
    rows = []
    attempted = {"candidate": 0, "predict": 0, "metric": 0}
    completed = {"candidate": 0, "predict": 0, "metric": 0}
    event({"kind": "begin", "policy": policy, "weights": list(weights),
           "quotas": list(quotas), "panel": [list(x) for x in PANEL]})
    try:
        for group, index in PANEL:
            attempted["candidate"] += 1
            event({"kind": "candidate_attempt", "group": group, "index": index})
            item = evaluate(group, index)
            if type(item) is not dict or set(item) != {"cost", "predict_calls"}:
                raise ValueError("candidate receipt")
            cost, calls = item["cost"], item["predict_calls"]
            if type(cost) not in (int, float) or not math.isfinite(cost) or type(calls) is not int or calls != 2:
                raise ValueError("H2 candidate receipt")
            attempted["predict"] += calls
            completed["predict"] += calls
            attempted["metric"] += 1
            completed["metric"] += 1
            completed["candidate"] += 1
            row = {"group": group, "index": index, "cost": cost, "eligible": index < quotas[group]}
            rows.append(row)
            event({"kind": "candidate_return", **row, "predict_calls": calls})
        winner = min((r for r in rows if r["eligible"]), key=lambda r: (r["cost"], r["group"], r["index"]))
        result = {"status": "complete_unverified", "rows": rows, "winner": dict(winner),
                  "attempted": attempted, "completed": completed, "expected": dict(EXPECTED),
                  "actual_backend_accounting_verified": False, "scientific_pass": False}
        event({"kind": "end", "result": result})
        return result
    except BaseException as exc:
        event({"kind": "failure", "type": type(exc).__name__, "message": str(exc),
               "attempted_candidates": attempted["candidate"], "completed_candidates": completed["candidate"],
               "inflight_predict_and_metric": "unknown", "retry_authorized": False})
        raise
