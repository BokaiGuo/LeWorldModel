"""Bind a future matched-feedback host run to the frozen measured-work contract.

Only run() invokes the existing host runner. Plan and consume use saved assets.
External coordination and pre-run timestamp claims are operator assertions,
not automatically authenticated.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from experiments.spectral_reachability import p3_feedback_budget_contract_20261030_v1 as budget
from experiments.spectral_reachability import p3_matched_order_feedback_host_v2_codex_20261029 as host

NAME = "p3_feedback_online_binding_20261030_v1"
CONTRACT_SHA = "5781d953d05477e633659b3e9bb58d630ea59447720964af4e8678829f565b4f"
PREPARED = host.ROOT / "results/p3_matched_order_feedback_host_v2_codex_20261029_prepare_MAIN01.json"
CONTRACT = host.ROOT / "results/p3_feedback_budget_contract_20261030_v1_FREEZE01.json"
RESULTS = host.ROOT / "results"
ACK = "I_CONFIRMED_EXTERNAL_PRERUN_RECORD_AND_RESOURCE_COORDINATION"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def target(path, prefix):
    p = Path(path)
    check(p.is_absolute() and p.parent == RESULTS and p.resolve() == p and p.name.startswith(prefix),
          "output must be a canonical direct results path with the required prefix")
    return p


def absent(path):
    check(not path.exists() and not path.is_symlink(), "output already exists")


def verify_inputs():
    frozen = budget.read_pinned(CONTRACT, CONTRACT_SHA)
    check(frozen == budget.contract(), "budget contract or verifier drift")
    host.verify(PREPARED, budget.PREPARE_SHA)
    return frozen


def make_plan(run_path, receipt_path, analysis_path):
    verify_inputs()
    run_path = target(run_path, host.NAME + "_RUN_")
    receipt_path = target(receipt_path, host.NAME + "_receipt_")
    analysis_path = target(analysis_path, NAME + "_analysis_")
    paths = (run_path, receipt_path, analysis_path)
    check(len(set(paths)) == 3, "output paths must differ")
    for path in paths:
        absent(path)
    return {"schema": NAME + ".plan", "binding_source_sha256": budget.sha(__file__),
            "prepare_path": str(PREPARED), "prepare_sha256": budget.PREPARE_SHA,
            "contract_path": str(CONTRACT), "contract_sha256": CONTRACT_SHA,
            "run_path": str(run_path), "receipt_path": str(receipt_path), "analysis_path": str(analysis_path),
            "roster": [list(x) for x in budget.ROSTER], "scientific_pass": False,
            "external_prerun_timestamp_verified": False,
            "failure_rule": "original host lifecycle; preserve all slots; no retry or padding"}


def verify_plan(path, expected):
    plan = budget.read_pinned(path, expected)
    check(type(plan) is dict and plan.get("schema") == NAME + ".plan"
          and plan.get("binding_source_sha256") == budget.sha(__file__)
          and plan.get("prepare_path") == str(PREPARED) and plan.get("prepare_sha256") == budget.PREPARE_SHA
          and plan.get("contract_path") == str(CONTRACT) and plan.get("contract_sha256") == CONTRACT_SHA
          and plan.get("roster") == [list(x) for x in budget.ROSTER]
          and plan.get("scientific_pass") is False and plan.get("external_prerun_timestamp_verified") is False
          and plan.get("failure_rule") == "original host lifecycle; preserve all slots; no retry or padding",
          "plan identity or source drift")
    run_path = target(plan["run_path"], host.NAME + "_RUN_")
    receipt_path = target(plan["receipt_path"], host.NAME + "_receipt_")
    analysis_path = target(plan["analysis_path"], NAME + "_analysis_")
    check(len({run_path, receipt_path, analysis_path}) == 3, "output paths must differ")
    verify_inputs()
    return plan


def coordination(plan_sha, reference):
    check(type(reference) is str and 8 <= len(reference.strip()) <= 1000 and reference == reference.strip()
          and "\n" not in reference and "\r" not in reference, "external coordination reference required")
    return json.dumps({"plan_sha256": plan_sha, "external_coordination_reference": reference,
                       "prerun_assertion": "operator_asserted_not_verified"},
                      sort_keys=True, separators=(",", ":"))


def run(plan_path, plan_sha, reference, authorization, acknowledgment):
    check(authorization == host.AUTH and acknowledgment == ACK, "explicit authorization and human acknowledgment required")
    plan = verify_plan(plan_path, plan_sha)
    for key in ("run_path", "receipt_path", "analysis_path"):
        absent(Path(plan[key]))
    record = coordination(plan_sha, reference)
    # The host runner independently re-verifies sources and applies its own guards.
    return host.run(PREPARED, budget.PREPARE_SHA, plan["run_path"], authorization, record)


def require_run_binding(plan, plan_sha, reference, report_sha, completion_sha):
    directory = Path(plan["run_path"])
    check(directory.is_dir() and not directory.is_symlink() and directory.resolve() == directory, "run directory")
    report = budget.read_pinned(directory / "report.json", report_sha)
    completion = budget.read_pinned(directory / "completion.json", completion_sha)
    check(report.get("coordination") == coordination(plan_sha, reference)
          and report.get("prepare_sha256") == budget.PREPARE_SHA
          and report.get("scientific_pass") is False
          and completion.get("report_sha256") == report_sha
          and completion.get("scientific_pass") is False, "run-plan binding")
    return report


def consume(plan_path, plan_sha, reference, report_sha, completion_sha):
    plan = verify_plan(plan_path, plan_sha)
    absent(Path(plan["receipt_path"]))
    absent(Path(plan["analysis_path"]))
    require_run_binding(plan, plan_sha, reference, report_sha, completion_sha)
    from experiments.spectral_reachability import p3_matched_order_feedback_host_saved_v2_codex_20261029 as saved
    receipt = saved.consume(plan["run_path"], report_sha, completion_sha)
    # Partial receipts are durable, but not treated as complete.
    receipt_sha = host.write_new(plan["receipt_path"], receipt)
    return {"receipt_sha256": receipt_sha, "status": receipt["status"], "scientific_pass": False}


def validate_full_ledger(ledger):
    host.life.seam.core.audit_ledger(ledger)
    check(ledger["caps"] == budget.EXPECTED, "numerical common cap mismatch")
    check(ledger["attempted"] == ledger["completed"] == budget.EXPECTED, "numerical actual budget mismatch")


def settle_full_run(plan, report):
    """Check paid ledger ceilings and expose actual physical/encoding counts.

    The strict consumer is responsible for native resource census and event
    authenticity; this adds the common-cap binding and does not sum axes.
    """
    from experiments.spectral_reachability import p3_matched_order_feedback_numeric_bridge_codex_20261029 as bridge

    directory = Path(plan["run_path"])
    journal = directory / "events.jsonl"
    check(budget.sha(journal) == report["journal_sha256"], "settlement journal hash")
    chunks, meters, encoding = [], [Counter() for _ in range(4)], [[] for _ in range(4)]
    slot = None
    with journal.open() as f:
        for line in f:
            event = json.loads(line)
            name, detail = event["resource"], event["detail"]
            if name == "matched_slot_attempt":
                slot = detail["slot"]
            elif name == "matched_slot_return":
                slot = None
            elif name == "matched_numeric_chunk":
                chunks.append(event)
            elif slot is not None and name.startswith("native_meter/") and event["kind"] == "intent":
                meters[slot][name.removeprefix("native_meter/")] += detail["units"]
            elif slot is not None and name == "encoding_complete":
                encoding[slot].append(detail["boundary"])
    slots = []
    for i in range(4):
        wrapped = bridge.reconstruct(chunks, i)
        finals = [v["payload"]["result"] for v in wrapped if v["payload"].get("kind") == "slot_final"]
        check(len(finals) == 1 and host.life.seam.core.digest(finals[0]) == report["rows"][i]["result_sha256"],
              "settlement slot identity")
        ledger = finals[0]["resource_receipt"]
        validate_full_ledger(ledger)
        check(encoding[i] == [5, 10, 15, 20], "encoding boundary settlement")
        slots.append({"slot": i, "numerical_caps": ledger["caps"], "numerical_attempted": ledger["attempted"],
                      "numerical_completed": ledger["completed"], "native_meter_attempted": dict(sorted(meters[i].items())),
                      "observation_encoding_boundaries": encoding[i], "slot_wall_ns": None,
                      "cuda_ns": ledger["cuda_ns"], "FLOPs": ledger["FLOPs"]})
    pairs = [{"root": root, "native_meter_vector_match":
              slots[2 * root]["native_meter_attempted"] == slots[2 * root + 1]["native_meter_attempted"]}
             for root in range(2)]
    return {"slots": slots, "pairs": pairs, "equal_FLOPs": False,
            "limitation": "observed instrumented units only, not equal total compute or wall time"}


def inspect(plan_path, plan_sha, reference, receipt_sha):
    plan = verify_plan(plan_path, plan_sha)
    absent(Path(plan["analysis_path"]))
    receipt = budget.read_pinned(plan["receipt_path"], receipt_sha)
    report = require_run_binding(plan, plan_sha, reference, receipt["report_sha256"], receipt["completion_sha256"])
    # A caller-chosen hash authenticates bytes, not who created a receipt.
    # Reconstruct the authoritative saved receipt before using it.
    from experiments.spectral_reachability import p3_matched_order_feedback_host_saved_v2_codex_20261029 as saved
    rebuilt = saved.consume(plan["run_path"], receipt["report_sha256"], receipt["completion_sha256"])
    check(rebuilt == receipt, "strict receipt reconstruction mismatch")
    check(type(receipt.get("full_saved_strict")) is bool, "strict receipt flag type")
    settlement = settle_full_run(plan, report) if receipt["full_saved_strict"] else None
    analysis = budget.inspect(plan["run_path"], plan["receipt_path"], receipt_sha,
                              PREPARED, CONTRACT, CONTRACT_SHA)
    analysis["online_resource_settlement"] = settlement
    analysis["online_binding"] = {"plan_sha256": plan_sha, "reference": reference,
                                  "external_prerun_timestamp_verified": False}
    output_sha = budget.save_new(plan["analysis_path"], analysis)
    return {"analysis_sha256": output_sha, "status": analysis["status"], "scientific_pass": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    for x in ("run", "receipt", "analysis", "output"):
        p.add_argument("--" + x, required=True)
    for cmd in ("run", "consume", "inspect"):
        p = sub.add_parser(cmd)
        for x in ("plan", "plan-sha", "reference"):
            p.add_argument("--" + x, required=True)
        if cmd == "run":
            p.add_argument("--authorization", required=True)
            p.add_argument("--acknowledgment", required=True)
        elif cmd == "consume":
            p.add_argument("--report-sha", required=True)
            p.add_argument("--completion-sha", required=True)
        else:
            p.add_argument("--receipt-sha", required=True)
    a = ap.parse_args()
    if a.command == "plan":
        output = target(a.output, NAME + "_plan_")
        result = {"plan_sha256": budget.save_new(output, make_plan(a.run, a.receipt, a.analysis))}
    elif a.command == "run":
        result = run(a.plan, a.plan_sha, a.reference, a.authorization, a.acknowledgment)
    elif a.command == "consume":
        result = consume(a.plan, a.plan_sha, a.reference, a.report_sha, a.completion_sha)
    else:
        result = inspect(a.plan, a.plan_sha, a.reference, a.receipt_sha)
    print(json.dumps(result, sort_keys=True))
    if a.command == "run" and result["status"] != "four_complete_candidate":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
