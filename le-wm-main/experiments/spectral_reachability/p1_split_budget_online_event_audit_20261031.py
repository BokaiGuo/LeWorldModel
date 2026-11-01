"""Audit hash-chained live feedback events and compare sealed offline executions."""
import argparse
import hashlib
import json
from pathlib import Path

from . import p1_split_budget_online_20261031 as online
from . import codex_p1_fixed_suffix_censored_consume_20261030 as journal

ROOT, need, sha = online.ROOT, online.need, online.sha
NAME = "p1_split_budget_online_event_audit_20261031"
OFFLINE = ROOT / "results/p1_split_budget_feedback_physical_20261031_run_RUN01"
OFFLINE_REPORT_SHA = "b436bd183e3d27ba382f8ad92ae33e26adc4b8dddaac91ff599f062197ece5a3"
PLAN = online.split.PLAN
PLAN_SHA = online.split.PLAN_SHA


def audit(directory, run_sha, prepared, prep_sha, consumed, consumed_sha):
    directory, prepared, consumed = Path(directory), Path(prepared), Path(consumed)
    need(sha(prepared) == prep_sha and sha(directory / "run_report.json") == run_sha
         and sha(consumed) == consumed_sha and sha(OFFLINE / "run_report.json") == OFFLINE_REPORT_SHA
         and sha(PLAN) == PLAN_SHA, "input hash")
    fixed = json.loads(prepared.read_bytes())
    need(fixed == online.build(), "prepared rebuild")
    report = json.loads((directory / "run_report.json").read_bytes())
    offline = json.loads((OFFLINE / "run_report.json").read_bytes())
    checked = json.loads(consumed.read_bytes())
    plan = json.loads(PLAN.read_bytes())
    expected_model_hash = online.planner.midpoint.terminal.curve.planner.adam.prior.base.MODEL_HASH
    need(report["host"]["device"] == "cuda:0"
         and report["model_before"] == report["model_after"]
         and report["model_before"]["original_state_dict"] == expected_model_hash["original_state_dict"]
         and report["model_before"]["all_buffers"] == expected_model_hash["all_buffers"],
         "persisted host and frozen model hashes")
    need(report["status"] == offline["status"] == "recorded_not_consumed"
         and report["prepare_sha256"] == prep_sha
         and len(report["shards"]) == len(offline["shards"]) == len(checked["rows"]) == 24,
         "complete roster")
    rows = []
    for i, (item, shard, old_shard, check, planned) in enumerate(zip(
            fixed["roster"], report["shards"], offline["shards"], checked["rows"], plan["rows"], strict=True)):
        need(item["index"] == shard["index"] == old_shard["index"] == check["index"] == planned["index"] == i,
             "index")
        path, oldpath = directory / f"row_{i:02d}" / "report.json", OFFLINE / f"row_{i:02d}" / "report.json"
        need(sha(path) == shard["report_sha256"] and sha(oldpath) == old_shard["report_sha256"], "shard hashes")
        current, old = json.loads(path.read_bytes()), json.loads(oldpath.read_bytes())
        events = journal.events(path.parent / "journal.jsonl", current["journal_chain_tail"])
        plans = [e for e in events if e["kind"] == "online_midpoint_plan"]
        observations = [e for e in events if e["kind"] == "fresh_observation"
                        and e["receipt"]["phase"] == "probe" and e["receipt"]["step"] == 5]
        next_steps = [e for e in events if e["kind"] == "returned_transition"
                      and e["phase"] == "probe" and e["step"] == 6]
        need(len(plans) == 1 and len(observations) == 3 and len(next_steps) == 1, "event roster")
        e = plans[0]
        need(e["anchor_index"] == i and e["role"] == item["role"]
             and e["pixel_sha256"] == item["midpoint"]["pixel_sha256"]
             and e["second_packet_sha256"] == hashlib.sha256(bytes.fromhex(item["raw10_float32_hex"])[40:]).hexdigest()
             and e["winner"] == planned["winner"] and e["scores"] == planned["scores"]
             and e["calls_code_path"] == {"encode": 2, "predict": 44, "action_encoder": 44, "backward": 40}
             and max(x["index"] for x in observations) < e["index"] < next_steps[0]["index"], "event binding/order")
        rows.append({"index": i, "role": item["role"], "event_index": e["index"],
                     "offline_row_identical": current["row"] == old["row"],
                     "offline_receipts_identical": current["receipts"] == old["receipts"],
                     "status": check["status"], "returned_suffix": check["returned_suffix"],
                     "native_success": check["last_accepted_state_metrics"]["native_success"],
                     "synchronized_seconds": e["synchronized_seconds"]})
    need(all(sha(p) == h for p, h in fixed["bindings"].items()), "final source drift")
    return {"schema": NAME, "status": "journal_checks_passed", "rows": rows,
            "input_report_sha256": run_sha, "input_consume_sha256": consumed_sha,
            "P1_admission": False, "independent_confirmation": False, "equal_budget_certified": False,
            "limitations": "Known roots; parity guard; no independent confirmation or equal-budget claim."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("directory", "run-sha", "prepared", "prep-sha", "consumed", "consumed-sha", "output"):
        p.add_argument("--" + option, required=True)
    a = p.parse_args()
    out = Path(a.output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    value = audit(a.directory, a.run_sha, a.prepared, a.prep_sha, a.consumed, a.consumed_sha)
    print(json.dumps({"rows": len(value["rows"]), "sha256": online.physical.design.run._write_new(out, value)}))
