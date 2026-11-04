"""Saved pre-outcome controller chronology and resource audit, no model execution."""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import p3_preoutcome_classified_control_20261102_codexexclusive as ctl
from . import codex_p3_quota_journal_20261030_v1 as host_journal


def read(path):
    raw = Path(path).read_bytes()
    ctl.require(raw and raw.endswith(b"\n") and len(raw) < 16 * 1024 * 1024, "controller journal truncated/oversize")
    previous, events = None, []
    for index, line in enumerate(raw.splitlines()):
        event = json.loads(line)
        ctl.require(event["index"] == index and event["previous_sha256"] == previous, "journal chronology")
        ctl.require(event["sha256"] == ctl.digest({k: v for k, v in event.items() if k != "sha256"}), "journal hash")
        previous = event["sha256"]
        events.append(event)
    return events


def verify_events(events, plan, bank):
    pos = 0
    attempted = {arm: dict.fromkeys(ctl.AXES, 0) for arm in ctl.ARMS}
    returned = {arm: dict.fromkeys(ctl.AXES, 0) for arm in ctl.ARMS}
    phases = {arm: {phase: {"attempted": dict.fromkeys(ctl.AXES, 0),
                            "returned": dict.fromkeys(ctl.AXES, 0)}
                    for phase in ctl.PHASES} for arm in ctl.ARMS}

    def take(kind, **match):
        nonlocal pos
        ctl.require(pos < len(events), "incomplete controller")
        event = events[pos]
        pos += 1
        ctl.require(event["kind"] == kind and all(event.get(k) == v for k, v in match.items()),
                    f"unexpected event at {pos-1}: expected {kind}")
        return event

    def charge(kind, arm, phase, root, charge):
        take(kind, arm=arm, phase=phase, root=root, charge=charge)
        record = attempted if kind == "call_intent" else returned
        label = "attempted" if kind == "call_intent" else "returned"
        for k, v in charge.items():
            record[arm][k] += v
            phases[arm][phase][label][k] += v
            ctl.require(record[arm][k] <= plan["caps_per_arm"][k], "cap exceeded")

    def forward(arm, phase, root):
        for charge_value in ({"action_encode": 1}, {"predict_calls": 1, "predict_samples": 1}) * 2:
            charge("call_intent", arm, phase, root, charge_value)
            charge("call_returned", arm, phase, root, charge_value)

    first = take("plan_frozen")
    ctl.require(first["plan"] == plan and first["plan_sha256"] == ctl.digest(plan)
                and plan["bank_sha256"] == ctl.digest(bank), "plan/bank binding")
    signals = {}
    for arm in ctl.ARMS:
        signals[arm] = []
        for root in (0, 1):
            charge("call_intent", arm, "signal", root, {"signal_compute": 1})
            forward(arm, "signal", root)
            charge("call_returned", arm, "signal", root, {"signal_compute": 1})
            value = take("signal_observed", arm=arm, root=root)["value"]
            ctl.require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid signal")
            signals[arm].append(value)
    ctl.require(signals["signal"] == signals["reverse"] and signals["signal"][0] != signals["signal"][1],
                "signal pairing")
    major, minor = ctl.QUOTAS[plan["point"]]
    allocation = [major, minor] if signals["signal"][0] > signals["signal"][1] else [minor, major]
    allocation = {"signal": allocation, "reverse": list(reversed(allocation))}
    take("both_allocations_frozen", signals=signals, allocations=allocation)
    winners = {}
    for arm in ctl.ARMS:
        winners[arm] = []
        for root, count in enumerate(allocation[arm]):
            values = []
            for index in range(count):
                take("candidate_intent", arm=arm, root=root, candidate_index=index,
                     candidate_sha256=ctl.digest(bank[root][index]))
                charge("call_intent", arm, "search", root, {"candidate_evaluate": 1})
                forward(arm, "search", root)
                charge("call_returned", arm, "search", root, {"candidate_evaluate": 1})
                score = take("candidate_returned", arm=arm, root=root, candidate_index=index)["score"]
                ctl.require(type(score) in (int, float) and math.isfinite(score), "invalid score")
                values.append((score, index))
            score, index = min(values)
            winners[arm].append({"root": root, "index": index, "score": score})
    result = take("pair_completed")["result"]
    ctl.require(pos == len(events) and result["allocations"] == allocation and result["winners"] == winners,
                "result/winner mismatch")
    for arm in ctl.ARMS:
        resource = result["resources"][arm]
        ctl.require(resource["attempted"] == attempted[arm] and resource["returned"] == returned[arm]
                    and resource["by_phase"] == phases[arm] and resource["caps"] == plan["caps_per_arm"]
                    and resource["failed"] is False, "resource accounting")
    return {"status": "saved_controller_consistency_known_roots", "allocation": allocation,
            "winners": winners, "attempted": attempted, "signals_recorded": signals,
            "signal_or_scores_recomputed_from_model": False, "physical_certified": False,
            "independent_outcome": False, "equal_total_compute": False,
            "P3_admission": False, "C3_admission": False}


def consume(run, report_sha, prepare_sha, verify_prepare):
    from . import codex_p3_online_budget_host_20261030_v1 as host
    run = Path(run)
    ctl.require(run.is_absolute() and run.parent == host.ROOT / "results"
                and run.name.startswith("p3_preoutcome_cuda_host_20261102_codexexclusive_RUN_")
                and run.is_dir() and not run.is_symlink(), "invalid run directory")
    ctl.require(host.prior.sha(run / "report.json") == report_sha, "external report SHA")
    plan, _, bank = verify_prepare(run / "preoutcome_prepare.json", prepare_sha)
    report = json.loads((run / "report.json").read_bytes())
    ctl.require(report["status"] == "panel_completed_unverified"
                and report["prepare_sha256"] == host.prior.sha(run / "prepare.json")
                and json.loads((run / "prepare.json").read_bytes()) == plan["host_plan"]
                and report["model_before"] == report["model_after"] == plan["host_plan"]["expected_model_hash"]
                and report["gpu_snapshot"]["uuid"] == host.UUID, "host report identity")
    events = host_journal.read(run / "events.jsonl", report["journal_tail"], host.CAP)
    ctl.require(len(events) == 1 and events[0]["kind"] == "preoutcome_controller_completed",
                "host event completion")
    ctl.require(host.prior.sha(run / "controller.jsonl") == events[0]["controller_sha256"],
                "controller file binding")
    controller_events = read(run / "controller.jsonl")
    result = verify_events(controller_events, plan["schedule"], bank)
    ctl.require(controller_events[-1]["result"] == events[0]["result"], "host result binding")
    result["report_sha256"] = report_sha
    result["prepare_sha256"] = prepare_sha
    return result
