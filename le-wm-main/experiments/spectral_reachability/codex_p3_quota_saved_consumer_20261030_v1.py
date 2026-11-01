"""Independent semantic verification of a complete saved fixed-panel journal.

Not a source/host identity certification: caller must independently verify the
manifest and externally pin its hash and journal tail. No neural inference.
"""
from collections import Counter
import math

from experiments.spectral_reachability import codex_p3_quota_journal_20261030_v1 as journal
from experiments.spectral_reachability import agent_p3_resume_20261027_frontend as source
from experiments.spectral_reachability import moving_history_budget_scheduler_cuda_20261027_v1 as native

POLICIES = ("uniform", "signal", "reverse")
AXES = ("action_encoder", "predict", "predictor", "pred_proj")


def verify_events(events, tasks, banks, weights):
    if len(tasks) != 2 or len(banks) != 2 or len(weights) != 2:
        raise ValueError("fixed inputs")
    if any(type(w) not in (int, float) or not math.isfinite(w) for w in weights):
        raise ValueError("weights")
    if weights[0] == weights[1]:
        signal = (10, 10)
    else:
        signal = (15, 5) if weights[0] > weights[1] else (5, 15)
    allocations = {"uniform": (10, 10), "signal": signal,
                   "reverse": signal if signal == (10, 10) else signal[::-1]}
    if not events or events[0].get("kind") != "fixed_roster" or events[-1].get("kind") != "panel_complete":
        raise ValueError("complete terminal and fixed roster")
    head = events[0]
    if (head["policies"] != list(POLICIES) or
        head["allocations"] != {k: list(v) for k, v in allocations.items()} or
        head["candidate_order"] != [[g, i] for g in (0, 1) for i in range(15)] or
        head["task_hashes"] != [source.digest(t) for t in tasks] or
        head["bank_hashes"] != [source.digest(b[:15]) for b in banks]):
        raise ValueError("roster identity")
    active, entries, completed, rows, arm_events = None, {}, set(), [], []
    position = 0
    for event in events[1:-1]:
        kind = event.get("kind")
        if kind == "candidate_attempt":
            if active is not None or position >= 90 or len(arm_events) != position // 30:
                raise ValueError("candidate sequence")
            policy = POLICIES[position // 30]
            group, index = (position % 30) // 15, position % 15
            scope = [policy, group, index]
            item = banks[group][index]
            if event["scope"] != scope or event["candidate_sha256"] != source.digest(item):
                raise ValueError("candidate identity")
            active = (scope, len(entries))
        elif kind == "resource_attempt":
            e = event["entry"]
            if active is None or e["scope"] != active[0] or event["scope"] != active[0] or e["index"] != len(entries) or e["completed"] is not False or e["resource"] not in AXES:
                raise ValueError("resource attempt")
            entries[e["index"]] = e
        elif kind == "resource_return":
            e = event["entry"]
            index = e["index"]
            if active is None or index not in entries or index in completed or event["scope"] != active[0]:
                raise ValueError("resource return")
            original = entries[index]
            if (e != dict(original, completed=True, ended_seconds=e.get("ended_seconds")) or
                type(e["ended_seconds"]) not in (int, float) or not math.isfinite(e["ended_seconds"]) or
                type(original["started_seconds"]) not in (int, float) or not math.isfinite(original["started_seconds"]) or
                e["ended_seconds"] < original["started_seconds"]):
                raise ValueError("resource return identity")
            completed.add(index)
        elif kind == "candidate_complete":
            if active is None:
                raise ValueError("candidate without attempt")
            row = event["row"]
            policy, group, index = active[0]
            item = banks[group][index]
            subset = [entries[i] for i in range(active[1], len(entries))]
            if (event["scope"] != active[0] or
                (row["policy"], row["group"], row["index"]) != (policy, group, index) or
                row["candidate_sha256"] != source.digest(item) or row["commands"] != item["commands"] or
                row["eligible"] is not (index < allocations[policy][group]) or
                row["entry_start"] != active[1] or row["entry_end"] != len(entries) or
                Counter(e["resource"] for e in subset) != Counter({k: 2 for k in AXES}) or
                any(e["index"] not in completed for e in subset)):
                raise ValueError("candidate receipt")
            native.check_path(tasks[group], row["commands"], row["path"])
            if row["cost"] != native.cost(tasks[group], row["path"]):
                raise ValueError("candidate cost")
            rows.append(row)
            position += 1
            active = None
        elif kind == "arm_complete":
            if active is not None or position not in (30, 60, 90) or len(arm_events) != position // 30 - 1:
                raise ValueError("arm order")
            policy = POLICIES[position // 30 - 1]
            group_rows = [r for r in rows if r["policy"] == policy]
            winners = [min((r for r in group_rows if r["group"] == g and r["eligible"]),
                           key=lambda r: (r["cost"], r["index"])) for g in (0, 1)]
            if event["policy"] != policy or event["winners"] != winners:
                raise ValueError("winner reconstruction")
            arm_events.append(event)
        elif kind not in ("module_input", "module_output", "context_input", "predict_output", "native_packet"):
            raise ValueError("unknown event")
        elif active is None or event["scope"] != active[0]:
            raise ValueError("native event scope")
    if position != 90 or len(arm_events) != 3 or len(entries) != 720 or len(completed) != 720:
        raise ValueError("complete panel census")
    if events[-1].get("arm_count") != 3 or events[-1].get("scientific_pass") is not False:
        raise ValueError("terminal")
    return {"status": "semantic_panel_checked_not_certified", "candidates": position,
            "operation_entries": len(entries), "winners": [e["winners"] for e in arm_events],
            "module_arrays_independently_reconstructed": False, "source_authenticated": False,
            "physical_certified": False, "scientific_pass": False}


def consume(path, tail, max_bytes, tasks, banks, weights):
    return verify_events(journal.read(path, tail, max_bytes), tasks, banks, weights)
