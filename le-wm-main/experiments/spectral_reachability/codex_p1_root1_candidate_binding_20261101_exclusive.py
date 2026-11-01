"""Reconstruct known root1 candidates from saved, pre-dispatch node records only."""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_four_arm_consumer_20261030 as consumer
from . import codex_p1_three_prefix_existing_audit_20261101_exclusive as prior
from . import codex_p1_cp10_robust_feedback_20261101_exclusive as robust
from . import p1_moving_live_feedback_analysis_20261027_v2 as journal

ROOT = prior.ROOT
NAME = "codex_p1_root1_candidate_binding_20261101_exclusive"
REPORTS = {
    "selected": "be3c3bd3706b20c70ae2c8e59ab323dae2faffdfdc7eecfb6cbc9502d4513c80",
    "zero": "773b19eebfc88565b2114eed8fee10f1ce70bd20c42e0f311211b6ea14450c8a",
    "greedy": "e2f868be54abb33c85cedee93749f7d588217b85df588287034f779575d608b1",
}
IDENTITY = robust.IDENTITY


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arr(value, shape):
    return robust.actions(value, shape)


def check_prefix(event, role, prefix):
    need(event["role"] == role, "prefix role")
    bank = np.asarray(event["bank"], dtype=np.float32)
    need(bank.ndim == 4 and bank.shape[1:] == (3, 5, 2)
         and np.isfinite(bank).all() and (np.abs(bank) <= 1).all(), "bank")
    scores = np.asarray(event["stage_scores"], dtype=np.float64)
    need(scores.shape == (len(bank),) and np.isfinite(scores).all(), "stage scores")
    index = event["original_selected_index"]
    need(type(index) is int and 0 <= index < len(bank), "selected index")
    plan = arr(event["full_selected"], (3, 5, 2))
    expected = (np.zeros_like(plan) if role == "zero" else
                bank[int(np.argmin(scores))] if role == "greedy" else bank[index])
    need(np.array_equal(plan, expected) and np.array_equal(plan[0], arr(prefix, (5, 2))),
         "role selection/prefix")
    return plan.tolist()


def bind_inputs(entry, actual_cp5, actual_prefix):
    need(all(actual_cp5.get(k) == v for k, v in entry["required_cp5"].items()), "cp5 identity")
    need(np.array_equal(arr(actual_prefix, (5, 2)), arr(entry["prefix"], (5, 2))), "prefix identity")
    return {"hold": arr(entry["hold"], (2, 5, 2)).tolist(),
            "optimized": arr(entry["optimized"], (2, 5, 2)).tolist()}


def one_event(rows, kind):
    matches = [r["payload"] for r in rows if r["kind"] == kind]
    need(len(matches) == 1, "trace event " + kind)
    return matches[0]


def check_terminal(rows):
    need(bool(rows) and rows[-1]["kind"] == "terminal"
         and rows[-1]["payload"]["status"] == "complete_node", "trace terminal")


def reconstruct(role):
    directory = ROOT / "results" / f"{prior.BASE}_root1_{role}_RUN_MAIN01"
    path = directory / "report.json"
    need(sha(path) == REPORTS[role], "pinned report")
    report = json.loads(path.read_bytes())
    prior_result = prior.audit_one(1, role)
    prepared = json.loads(Path(report["prepared"]).read_bytes())
    need(all(Path(p).is_file() and sha(p) == h for p, h in prepared["sources"].items()),
         "prepared source closure")
    jp = directory / "journal.jsonl"
    need(sha(jp) == report["journal_sha256"], "physical journal hash")
    journal.journal(jp.read_bytes(), report["journal_chain_tail"])
    row = report["rows"][0]
    checkpoints = {p["step"]: p for p in row["checkpoints"]}
    nodes = {n["boundary"]: n for n in report["nodes"]["s1_structured"]}
    events, receipts = {}, {}
    for boundary in (0, 5):
        receipt = nodes[boundary]["receipt"]
        raw = Path(receipt["trace"]).read_bytes()
        need(hashlib.sha256(raw).hexdigest() == receipt["trace_sha256"], "trace digest")
        events[boundary] = consumer._parse(raw)
        check_terminal(events[boundary])
        receipts[boundary] = receipt
    prefix_event = one_event(events[0], "prefix_intervention")
    original = check_prefix(prefix_event, role, row["executed_raw"][:5])
    need(one_event(events[0], "selected_full_plan")["raw"] == original, "b00 full plan")
    cp5_plan = one_event(events[5], "selected_full_plan")
    need(cp5_plan["boundary"] == 5, "cp5 boundary")
    optimized_full = arr(cp5_plan["raw"], (3, 5, 2))
    need(np.array_equal(optimized_full[0], arr(receipts[5]["commands"], (5, 2))), "cp5 seal commands")
    prefix = arr(original[0], (5, 2))
    hold = arr(original[1:], (2, 5, 2))
    optimized = arr(optimized_full[:2], (2, 5, 2))
    point = checkpoints[5]
    identity = {k: point[k] for k in IDENTITY}
    entry = {"prefix": prefix.tolist(), "hold": hold.tolist(), "optimized": optimized.tolist(),
             "prefix_float32_sha256": robust.digest_actions(prefix),
             "hold_float32_sha256": robust.digest_actions(hold),
             "optimized_float32_sha256": robust.digest_actions(optimized),
             "required_cp5": identity,
             "cp5_goal_state_sha256": journal.digest(point["goal"]),
             "source": {"report": str(path), "report_sha256": REPORTS[role],
                        "prepared": report["prepared"], "prepare_sha256": report["prepare_sha256"],
                        "journal_sha256": report["journal_sha256"],
                        "boundary0_trace": receipts[0]["trace"],
                        "boundary0_trace_sha256": receipts[0]["trace_sha256"],
                        "boundary5_trace": receipts[5]["trace"],
                        "boundary5_trace_sha256": receipts[5]["trace_sha256"]},
             "historical_node_charged_upper_bounds": {str(b): receipts[b]["account"]["charged_upper_bounds"]
                                                       for b in (0, 5)},
             "accounting_qualification": "closed_unverified; actual usage and coverage not independently certified",
             "prefix_selection": {"original_index": prefix_event["original_selected_index"],
                                  "greedy_index": int(np.argmin(prefix_event["stage_scores"])),
                                  "stage_scores": prefix_event["stage_scores"]}}
    bind_inputs(entry, identity, prefix)
    return entry, prior_result


def build():
    entries, inspections = {}, []
    for role in ("selected", "zero", "greedy"):
        entries[role], item = reconstruct(role)
        inspections.append(item)
    first = inspections[0]
    need(all(r["binding"] == first["binding"] and r["model"] == first["model"]
             and r["initial_state"] == first["initial_state"] and r["initial_pixel"] == first["initial_pixel"]
             and r["goal_state"] == first["goal_state"] for r in inspections), "root/goal/model mismatch")
    return {"schema": NAME + ".v1", "development_only": True, "root_slot": 1,
            "roles": ["selected", "zero", "greedy"], "goal_binding": first["binding"],
            "goal_state": first["goal_state"], "goal_state_sha256": journal.digest(first["goal_state"]),
            "model": first["model"], "root_pixel_sha256": first["initial_pixel"],
            "candidates": entries,
            "candidate_definition": {"hold": "boundary0 role-selected H3 plan packets 1:3; zero uses zero H3",
                                     "optimized": "own actual cp5 DLS H3 selected full plan packets 0:2",
                                     "time": "both candidates trace-sealed at or before own cp5 dispatch"},
            "method_difference": "root1 DLS H3 generation and truncation; not root0 diag32 candidate4 iteration13 or cp5 two-packet 32-step optimizer",
            "future_robust_scoring_budget_per_complete_episode": robust.protocol()["neural_caps_per_complete_episode"],
            "future_physical_execution_authorized_by_this_artifact": False,
            "source_contracts_verified_against_saved_hashes": True,
            "full_cost_fairness_verified": False, "outcomes_known_before_reconstruction": True,
            "independent_confirmation": False, "P1_admission": False, "new_neural_calls": 0,
            "new_physical_calls": 0,
            "audit_code_sha256": sha(Path(__file__)),
            "test_code_sha256": sha(ROOT / "tests" / ("test_" + NAME + ".py"))}


def main():
    target = ROOT / "results" / (NAME + "_CODEX01.json")
    value = build()
    with target.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(target), "sha256": sha(target)}))


if __name__ == "__main__":
    main()
