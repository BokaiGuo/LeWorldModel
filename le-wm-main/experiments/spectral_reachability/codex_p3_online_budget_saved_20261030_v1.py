"""Independent saved semantics for the online two-point panel, no model execution."""
from collections import Counter
import math
import numpy as np
from experiments.spectral_reachability import codex_p3_quota_journal_20261030_v1 as journal
from experiments.spectral_reachability import moving_history_budget_scheduler_cuda_20261027_v1 as native

POINTS, ARMS = (20, 30), ("uniform", "signal", "reverse")
NEURAL = ("action_encoder", "predict", "predictor", "pred_proj")
AD = ("jacobian", "vjp_directions", "grad_api", "raw_backward_hook", "svd")
AXES = NEURAL + AD + ("metric",)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify_events(events, tasks, banks):
    require(len(tasks) == len(banks) == 2 and all(len(b) == 30 for b in banks), "inputs")
    require(events and events[0]["kind"] == "experiment_started" and
            events[-1]["kind"] == "experiment_complete" and events[-1]["arms"] == 6 and
            events[-1]["scientific_pass"] is False, "complete boundaries")
    head = events[0]
    expected_caps = {str(n): {**{k: 2*n+8 for k in NEURAL}, "jacobian": 2,
                     "vjp_directions": 384, "grad_api": 384, "raw_backward_hook": 384,
                     "svd": 2, "metric": n+4} for n in POINTS}
    require(head["points"] == list(POINTS) and head["arms"] == list(ARMS) and
            head["caps"] == expected_caps and head["signal_sharing"] is False and
            head["bank_tag"] == "p3-online-budget-v1-cold-pcg64", "fixed roster")
    semantic = {"nominal_request", "signal_complete", "allocation_frozen", "candidate_attempt",
                "candidate_complete", "arm_complete"}
    allowed = semantic | {"resource_attempt", "resource_return", "module_input", "module_output",
                          "context_input", "predict_output", "native_packet", "jacobian_forward_packet",
                          "jacobian_row", "raw_backward"}
    require(all(e["kind"] in allowed for e in events[1:-1]), "unknown or failure event")
    entries, returns = {}, {}
    for event in events[1:-1]:
        if event["kind"] not in ("resource_attempt", "resource_return"):
            continue
        e = event["entry"]
        i = e["index"]
        require(type(i) is int and e["resource"] in AXES and e["scope"] == event["scope"], "ledger identity")
        if event["kind"] == "resource_attempt":
            require(i == len(entries) and e["completed"] is False and
                    type(e["started_seconds"]) in (float, int) and math.isfinite(e["started_seconds"]),
                    "ordered attempt")
            entries[i] = e
        else:
            require(i in entries and i not in returns and e == dict(entries[i], completed=True,
                    ended_seconds=e.get("ended_seconds")) and
                    type(e["ended_seconds"]) in (float, int) and math.isfinite(e["ended_seconds"]) and
                    e["ended_seconds"] >= e["started_seconds"], "matched return")
            returns[i] = e
    require(len(entries) == len(returns), "all operations returned")
    cursor = 1
    def section(kind, scope, vector=None):
        nonlocal cursor
        start = cursor
        while cursor < len(events)-1 and events[cursor]["kind"] not in semantic:
            require(events[cursor]["scope"] == scope, "event outside scoped section")
            cursor += 1
        require(cursor < len(events)-1 and events[cursor]["kind"] == kind and
                events[cursor]["scope"] == scope, "semantic sequence")
        segment = events[start:cursor]
        if vector is not None:
            counts = Counter(e["entry"]["resource"] for e in segment if e["kind"] == "resource_attempt")
            require(dict(counts) == vector, "segment resource vector")
        result = events[cursor]
        cursor += 1
        return result, segment
    result = []
    for point in POINTS:
        for arm in ARMS:
            weights = None
            if arm != "uniform":
                weights = []
                for root in (0, 1):
                    scope = [point, arm, "signal", root]
                    nominal, _ = section("nominal_request", scope, {})
                    require(nominal["commands"] == [[0.0, 0.0]] * 10, "zero nominal")
                    sv = {**{k: 4 for k in NEURAL}, "jacobian": 1, "vjp_directions": 192,
                          "grad_api": 192, "raw_backward_hook": 192, "svd": 1}
                    observed, segment = section("signal_complete", scope, sv)
                    native.check_path(tasks[root], nominal["commands"], observed["nominal_contexts"])
                    terminal = np.asarray(observed["terminal"], dtype=np.float64)
                    require(np.array_equal(terminal, observed["nominal_contexts"][-1]["latent_history"][-1]),
                            "signal nominal terminal")
                    jac = native.unblob(observed["jacobian"])
                    jr = [e for e in segment if e["kind"] == "jacobian_row"]
                    require(len(jr) == 192 and [e["row"] for e in jr] == list(range(192)) and
                            np.array_equal(jac, np.stack([native.unblob(e["array"]) for e in jr])),
                            "jacobian row reconstruction")
                    signal = native.signal(tasks[root], terminal, jac)
                    require(signal == observed["signal"], "signal reconstruction")
                    weights.append(signal["w"])
            if arm == "uniform":
                quota = [(point+4)//2] * 2
            elif weights[0] == weights[1]:
                quota = [point//2] * 2
            else:
                high = int(weights[1] > weights[0])
                if arm == "reverse":
                    high = 1-high
                amount = point*3//4
                quota = [amount, point-amount] if high == 0 else [point-amount, amount]
            alloc, _ = section("allocation_frozen", [point, arm, "search", 0], {})
            require(alloc["quota"] == quota and alloc["weights"] == weights, "online allocation")
            winners = []
            for root in (0, 1):
                scope = [point, arm, "search", root]
                candidates = []
                for index in range(quota[root]):
                    attempt, _ = section("candidate_attempt", scope, {})
                    require(attempt["index"] == index and attempt["commands"] == banks[root][index]["commands"],
                            "candidate request identity")
                    done, _ = section("candidate_complete", scope,
                                      {**{k: 2 for k in NEURAL}, "metric": 1})
                    row = done["candidate"]
                    require(row["index"] == index and row["commands"] == attempt["commands"], "candidate identity")
                    native.check_path(tasks[root], row["commands"], row["path"])
                    require(row["cost"] == native.cost(tasks[root], row["path"]), "candidate cost")
                    candidates.append(row)
                winners.append(min(candidates, key=lambda r: (r["cost"], r["index"])))
            final, _ = section("arm_complete", [point, arm, "search", 1], {})
            row = final["row"]
            counts = Counter(e["resource"] for e in entries.values() if e["scope"][:2] == [point, arm])
            expected = {k: counts[k] for k in AXES}
            require(row["point"] == point and row["arm"] == arm and row["weights"] == weights and
                    row["allocation"] == quota and row["winners"] == winners and
                    row["attempted"] == row["returned"] == expected and
                    all(expected[k] <= expected_caps[str(point)][k] for k in AXES), "arm summary")
            result.append(dict(point=point, arm=arm, allocation=quota, weights=weights,
                               attempted=expected, winners=winners))
    require(cursor == len(events)-1, "no extra semantic events")
    return dict(status="semantic_saved_checked_not_scientifically_admitted", rows=result,
                model_arrays_independently_recomputed=False, physical_certified=False, scientific_pass=False)


def consume(path, tail, cap, tasks, banks):
    return verify_events(journal.read(path, tail, cap), tasks, banks)
