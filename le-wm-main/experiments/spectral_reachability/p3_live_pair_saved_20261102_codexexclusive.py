"""Read-only reconstruction of a P3 live pair journal; no neural calls."""
from collections import Counter
import math

import numpy as np

from . import codex_p3_quota_journal_20261030_v1 as journal
from . import moving_history_budget_scheduler_cuda_20261027_v1 as native

ARMS = ("signal", "reverse")
SEMANTIC = {"nominal_request", "signal_complete", "allocation_frozen",
            "candidate_attempt", "candidate_complete", "arm_complete"}


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_events(events, tasks, banks, *, point=20):
    require(type(point) is int and point == 20 and len(tasks) == len(banks) == 2,
            "fixed development panel")
    require(all(len(bank) >= point for bank in banks), "candidate bank")
    attempts, returns = {}, {}
    for event in events:
        if event.get("kind") not in ("resource_attempt", "resource_return"):
            continue
        entry = event["entry"]
        index = entry["index"]
        require(type(index) is int and entry["scope"] == event["scope"], "resource scope")
        if event["kind"] == "resource_attempt":
            require(index == len(attempts) and entry["completed"] is False, "attempt order")
            attempts[index] = entry
        else:
            require(index in attempts and index not in returns and
                    entry == dict(attempts[index], completed=True,
                                  ended_seconds=entry.get("ended_seconds")) and
                    type(entry.get("ended_seconds")) in (float, int) and
                    math.isfinite(entry["ended_seconds"]) and
                    entry["ended_seconds"] >= entry["started_seconds"], "resource return")
            returns[index] = entry
    require(len(attempts) == len(returns), "incomplete resource attempts")
    cursor = 0

    def section(kind, scope, expected):
        nonlocal cursor
        start = cursor
        while cursor < len(events) and events[cursor]["kind"] not in SEMANTIC:
            require(events[cursor]["kind"] not in ("experiment_failure", "host_failure")
                    and events[cursor].get("scope") == scope, "unexpected section event")
            cursor += 1
        require(cursor < len(events) and events[cursor]["kind"] == kind and
                events[cursor]["scope"] == scope, "semantic sequence")
        segment = events[start:cursor]
        actual = Counter(e["entry"]["resource"] for e in segment if e["kind"] == "resource_attempt")
        require(dict(actual) == expected, "section resource vector")
        result = events[cursor]
        cursor += 1
        return result, segment

    neural = ("action_encoder", "predict", "predictor", "pred_proj")
    signal_vector = {**{key: 4 for key in neural}, "jacobian": 1,
                     "vjp_directions": 192, "grad_api": 192,
                     "raw_backward_hook": 192, "svd": 1}
    rows = []
    for arm in ARMS:
        weights = []
        for root in (0, 1):
            scope = [point, arm, "signal", root]
            initial, _ = section("nominal_request", scope, {})
            require(initial["commands"] == [[0.0, 0.0]] * 10, "nominal commands")
            observed, segment = section("signal_complete", scope, signal_vector)
            native.check_path(tasks[root], initial["commands"], observed["nominal_contexts"])
            terminal = np.asarray(observed["terminal"], dtype=np.float64)
            require(np.array_equal(terminal, observed["nominal_contexts"][-1]["latent_history"][-1]),
                    "signal terminal")
            jac = native.unblob(observed["jacobian"])
            jacrows = [e for e in segment if e["kind"] == "jacobian_row"]
            require(len(jacrows) == 192 and [e["row"] for e in jacrows] == list(range(192))
                    and np.array_equal(jac, np.stack([native.unblob(e["array"]) for e in jacrows])),
                    "jacobian rows")
            signal = native.signal(tasks[root], terminal, jac)
            require(signal == observed["signal"], "signal recomputation")
            weights.append(signal["w"])
        require(weights[0] != weights[1], "tied signals")
        high = int(weights[1] > weights[0])
        if arm == "reverse":
            high = 1 - high
        quota = [15, 5] if high == 0 else [5, 15]
        alloc, _ = section("allocation_frozen", [point, arm, "search", 0], {"scheduler": 1})
        require(alloc["quota"] == quota and alloc["weights"] == weights, "allocation")
        winners = []
        for root, count in enumerate(quota):
            candidates = []
            scope = [point, arm, "search", root]
            for index in range(count):
                attempt, _ = section("candidate_attempt", scope, {})
                require(attempt["index"] == index and
                        attempt["commands"] == banks[root][index]["commands"], "candidate request")
                done, _ = section("candidate_complete", scope,
                                  {**{key: 2 for key in neural}, "metric": 1})
                candidate = done["candidate"]
                require(candidate["index"] == index and candidate["commands"] == attempt["commands"],
                        "candidate identity")
                native.check_path(tasks[root], candidate["commands"], candidate["path"])
                require(candidate["cost"] == native.cost(tasks[root], candidate["path"]), "candidate cost")
                candidates.append(candidate)
            winners.append(min(candidates, key=lambda x: (x["cost"], x["index"])))
        final, _ = section("arm_complete", [point, arm, "search", 1], {})
        row = final["row"]
        counted = dict(Counter(e["resource"] for e in attempts.values() if e["scope"][:2] == [point, arm]))
        require(row["arm"] == arm and row["weights"] == weights and row["allocation"] == quota
                and row["winners"] == winners and row["attempted"] == row["returned"] == counted,
                "arm result")
        rows.append(row)
    require(cursor == len(events) and rows[0]["weights"] == rows[1]["weights"]
            and rows[0]["allocation"] == rows[1]["allocation"][::-1]
            and rows[0]["attempted"] == rows[1]["attempted"], "paired journal completion")
    return {"status": "saved_semantics_checked_known_roots", "point": point, "rows": rows,
            "model_outputs_recomputed": False, "equal_total_compute": False,
            "physical_certified": False, "independent_outcome": False,
            "P3_admission": False, "C3_admission": False}


def consume(path, tail, cap, tasks, banks):
    return verify_events(journal.read(path, tail, cap), tasks, banks)
