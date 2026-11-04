"""Fail-closed saved-output audit for the P2 joint/conditional host."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from . import p2_joint_vs_conditional_matched_20261102_codexexclusive as matched
from . import p2_common_objective_ledger_20261102_exclusive as core
from . import p2_same_bank_host_development_20261102_exclusive as base


def need(ok, message):
    if not ok:
        raise ValueError(message)


def audit_complete(path, result, bank=None):
    audit = core.verify_ledger(path)
    need(not audit["outstanding"], "outstanding calls")
    expected = {(category, axis): {"attempt": cap, "returned": cap, "failed": 0}
                for category, axes in matched.CAPS.items() for axis, cap in axes.items()}
    need(audit["counts"] == expected, "resource census")
    events = [json.loads(line)["record"] for line in Path(path).read_bytes().splitlines()]
    need(sum(e["kind"] == "matched_meter_complete" for e in events) == 1, "completion")
    need(not any(e["kind"] in ("matched_meter_reject", "matched_meter_incomplete", "optimizer_failed")
                 for e in events), "failed schedule")
    timings = [e for e in events if e["kind"] == "timed_call"]
    timed = Counter((e["category"], e["payload"]["axis"]) for e in timings)
    need(timed == Counter({k: v["attempt"] for k, v in expected.items()}), "timing census")
    need(all(e["payload"]["status"] == "returned" and e["payload"]["inclusive"] is True
             and type(e["payload"]["elapsed_ns"]) is int and e["payload"]["elapsed_ns"] >= 0
             for e in timings), "invalid timing")
    transitions = [e for e in events if e["kind"] == "transition_shape"]
    for category in matched.CAPS:
        records = [e["payload"] for e in transitions if e["category"] == category]
        n = matched.CAPS[category].get("init_transition", matched.CAPS[category].get("transition_packet"))
        need(len(records) == n and all(r == {"state_shape": [1, 3, 192], "history_shape": [1, 2, 10],
             "action_shape": [5, 2], "context_tokens": 3, "successor_tokens": 1} for r in records),
             "transition token census")
    rows = result["rows"]
    need(len(rows) == 16 and [r["candidate_index"] for r in rows] == list(range(16)), "bank census")
    if bank is not None:
        need(len(bank) == 16 and [r["source_sha256"] for r in rows]
             == [hashlib.sha256(raw).hexdigest() for raw in bank], "bank identity")
    starts = [e["payload"] for e in events if e["kind"] == "init_start"]
    initial = [e["payload"] for e in events if e["kind"] == "init_returned"]
    expected_order = [(i, arm) for i in range(16) for arm in matched.ARMS]
    need([(x["candidate"], x["arm"]) for x in starts] == expected_order
         and [(x["candidate"], x["arm"]) for x in initial] == expected_order, "initializer order")
    attempts = [e["payload"] for e in events if e["kind"] == "optimizer_attempt"]
    returned = [e["payload"] for e in events if e["kind"] == "optimizer_returned"]
    order = [(i, a, t) for i in range(16) for a in matched.ARMS for t in range(9)]
    key = lambda x: (x["candidate_index"], x["arm"], x["iteration"])
    need([key(x) for x in attempts] == order and [key(x) for x in returned] == order, "optimizer order")
    for a, b in zip(attempts, returned):
        need(a["action_sha256"] == b["action_sha256"] and
             a["gradient_required"] == b["gradient_required"] == (a["iteration"] < 8)
             and type(b["score"]) in (int, float) and math.isfinite(b["score"]), "optimizer receipt")
    for row in rows:
        i = row["candidate_index"]
        need([x["arm"] for x in row["arms"]] == list(matched.ARMS), "arm order")
        for arm, entry in zip(matched.ARMS, row["arms"]):
            j = i * 2 + matched.ARMS.index(arm)
            raw = bytes.fromhex(row["initializer_float32_hex"][arm])
            digest = hashlib.sha256(raw).hexdigest()
            need(len(raw) == 80 and digest == row["initializer_sha256"][arm] == initial[j]["output_sha256"]
                 and starts[j]["action_sha256"] == row["source_sha256"]
                 and initial[j]["jacobian_shape"] == [192, 10 if arm == matched.ARMS[0] else 20]
                 and initial[j]["requested_output_vjps"] == 192, "initializer binding")
            if bank is not None and arm == matched.ARMS[0]:
                need(raw[:40] == bank[i][:40], "conditional prefix changed")
            observations = [x for x in returned if x["candidate_index"] == i and x["arm"] == arm]
            best = min(observations, key=lambda x: (x["score"], x["iteration"]))
            winner_raw = bytes.fromhex(entry["raw_float32_hex"])
            need(len(winner_raw) == 80 and hashlib.sha256(winner_raw).hexdigest() == best["action_sha256"]
                 and entry["winner_iteration"] == best["iteration"] and entry["model_score"] == best["score"]
                 and observations[0]["action_sha256"] == digest
                 and all(entry[k] == v for k, v in (("attempted_score_calls", 9),
                     ("returned_score_calls", 9), ("attempted_gradients", 8),
                     ("returned_gradients", 8))), "winner binding")
    for arm in matched.ARMS:
        options = [(e["model_score"], r["candidate_index"], e["winner_iteration"], e)
                   for r in rows for e in r["arms"] if e["arm"] == arm]
        score, index, iteration, entry = min(options, key=lambda x: x[:3])
        need(result["model_selected_winners"][arm] == {"candidate_index": index, "iteration": iteration,
             "model_score": score, "raw_float32_hex": entry["raw_float32_hex"]}, "global winner")
    return {"ledger_tail_sha256": audit["tail_sha256"], "transition_calls": len(transitions),
            "observed_context_tokens": sum(e["payload"]["context_tokens"] for e in transitions),
            "observed_successor_tokens": sum(e["payload"]["successor_tokens"] for e in transitions),
            "timed_calls": len(timings), "gradient_calls": sum(v["returned"] for (c, a), v in expected.items()
                                                                 if a == "autograd_grad"),
            "jacobian_requested_output_directions": 32 * 192,
            "internal_backward_calls": None, "flops": None, "equal_total_compute": False}


def consume(folder, expected_report_sha):
    from . import p2_joint_host_20261102_codexexclusive as host
    folder = Path(folder)
    need(folder.is_absolute() and folder.parent == host.RESULTS and not folder.is_symlink()
         and folder.name.startswith(host.NAME + "_RUN_"), "run path")
    report_path = folder / "report.json"
    need(base.file_sha(report_path) == expected_report_sha, "report SHA")
    report = json.loads(report_path.read_bytes())
    binding = json.loads((folder / "binding.json").read_bytes())
    plan = host.verify_plan(binding["plan"], binding["plan_sha256"])
    need(report["plan_sha256"] == binding["plan_sha256"] and report["status"] == "completed_development"
         and report["failure"] is None and report["frozen_parameters_unchanged"]
         and report["equal_total_compute"] is False and report["P2_admission"] is False
         and report["device"]["uuid"] == host.UUID
         and base.file_sha(folder / "journal.jsonl") == report["journal_sha256"], "report binding")
    bank = host.native.verified_input()["bank"]
    need([base.sha(raw) for raw in bank] == plan["bank_sha256"], "current bank binding")
    summary = audit_complete(folder / "journal.jsonl", report["result"], bank)
    return {"schema": "p2_joint_saved_audit_v1", "report_sha256": expected_report_sha,
            "plan_sha256": binding["plan_sha256"], "summary": summary,
            "known_root_only": True, "historical_sra_authenticated": False,
            "independent_confirmation": False, "P2_admission": False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--report-sha256", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    from . import p2_joint_host_20261102_codexexclusive as host
    output = host.path_new(args.output, "_CONSUME_")
    value = consume(args.run, args.report_sha256)
    print(json.dumps({"output": str(output), "sha256": base.write_new(output, value)}))


if __name__ == "__main__":
    main()
