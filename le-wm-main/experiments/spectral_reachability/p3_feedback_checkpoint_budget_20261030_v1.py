"""CPU saved-only cumulative work at feedback checkpoints; not a producer."""
import argparse
import json
from collections import Counter
from pathlib import Path

from experiments.spectral_reachability import p3_feedback_budget_contract_20261030_v1 as budget
from experiments.spectral_reachability import p3_feedback_online_binding_20261030_v1 as binding
from experiments.spectral_reachability import p3_matched_order_feedback_host_saved_v2_codex_20261029 as saved

NAME = "p3_feedback_checkpoint_budget_20261030_v1"
CHECKPOINTS = (0, 5, 10, 15, 20)


def checkpoint_costs(events, wrapped, positions, slot):
    """Cumulative actual charges up to post-encoding boundaries (cp0 root accepted)."""
    start = [e["index"] for e in events if e["resource"] == "matched_slot_attempt" and e["detail"]["slot"] == slot]
    end = [e["index"] for e in events if e["resource"] == "matched_slot_return" and e["detail"]["slot"] == slot]
    if len(start) != 1 or len(end) != 1 or start[0] >= end[0] or len(wrapped) != len(positions):
        raise ValueError("slot event bounds")
    relevant = [e for e in events if start[0] < e["index"] < end[0]]
    root = [e["index"] for e in relevant if e["resource"] == "native_matched_root_accepted"]
    encoded = [e for e in relevant if e["resource"] == "encoding_complete"]
    if len(root) != 1 or [e["detail"]["boundary"] for e in encoded] != [5, 10, 15, 20]:
        raise ValueError("checkpoint markers")
    markers = {0: root[0], **{e["detail"]["boundary"]: e["index"] for e in encoded}}
    ops = []
    for value, position in zip(wrapped, positions, strict=True):
        if not (start[0] < position[0] <= position[1] < end[0]):
            raise ValueError("numeric position outside slot")
        payload = value["payload"]
        if payload["kind"] in ("resource_attempt", "resource_return", "resource_failure"):
            ops.append((position[1], payload["kind"], payload["entry"]))
    result = {}
    for step, cutoff in markers.items():
        attempted = dict.fromkeys(budget.EXPECTED, 0)
        completed = dict.fromkeys(budget.EXPECTED, 0)
        returned = []
        for position, kind, op in ops:
            if position > cutoff:
                continue
            if set(op["delta"]) != set(budget.EXPECTED):
                raise ValueError("resource axes")
            if kind == "resource_attempt":
                for k, v in op["delta"].items():
                    attempted[k] += v
            elif kind == "resource_return":
                for k, v in op["delta"].items():
                    completed[k] += v
                returned.append(op)
        physical = Counter()
        encoding = Counter()
        for e in relevant:
            if e["index"] > cutoff or e["kind"] != "intent":
                continue
            if e["resource"].startswith("native_meter/"):
                physical[e["resource"].removeprefix("native_meter/")] += e["detail"]["units"]
            elif e["origin"] == "encoding" and e["resource"] in ("encoder", "projector"):
                encoding[e["resource"]] += 1
        result[str(step)] = {"attempted": attempted, "completed": completed,
                             "hook_shape_counts": dict(sorted(budget.shapes(returned).items())),
                             "physical_meter_attempted": dict(sorted(physical.items())),
                             "observation_encoder_attempted": dict(sorted(encoding.items()))}
    return result


def compare(a, b):
    if a is None or b is None:
        return {str(t): {"match": None, "reason": "partial or missing slot"} for t in CHECKPOINTS}
    result = {}
    for t in CHECKPOINTS:
        left, right = a[str(t)], b[str(t)]
        result[str(t)] = {
            "numeric_attempted_match": left["attempted"] == right["attempted"],
            "numeric_completed_match": left["completed"] == right["completed"],
            "hook_shape_multiset_match": left["hook_shape_counts"] == right["hook_shape_counts"],
            "physical_meter_match": left["physical_meter_attempted"] == right["physical_meter_attempted"],
            "observation_encoder_match": left["observation_encoder_attempted"] == right["observation_encoder_attempted"],
            "predict_attempted_signal_minus_fixed": right["attempted"]["predict_API"] - left["attempted"]["predict_API"],
        }
    return result


def analyze(run, receipt_path, receipt_sha, prepared, contract_path, contract_sha,
            plan_path=None, plan_sha=None, reference=None):
    directory = Path(run).resolve()
    receipt = budget.read_pinned(receipt_path, receipt_sha)
    if any(x is not None for x in (plan_path, plan_sha, reference)):
        if not all(x is not None for x in (plan_path, plan_sha, reference)):
            raise ValueError("plan, hash and reference must be supplied together")
        plan = binding.verify_plan(plan_path, plan_sha)
        if directory != Path(plan["run_path"]) or Path(receipt_path).resolve() != Path(plan["receipt_path"]):
            raise ValueError("plan path mismatch")
        binding.require_run_binding(plan, plan_sha, reference, receipt["report_sha256"], receipt["completion_sha256"])
    original = saved.consume(str(directory), receipt["report_sha256"], receipt["completion_sha256"])
    if receipt != original:
        raise ValueError("strict receipt reconstruction mismatch")
    summary = budget.inspect(directory, receipt_path, receipt_sha, prepared, contract_path, contract_sha)
    out = {"schema": NAME + ".analysis", "receipt_sha256": receipt_sha,
           "report_sha256": receipt["report_sha256"], "contract_sha256": contract_sha,
           "plan_sha256": plan_sha, "external_prerun_timestamp_verified": False,
           "scientific_pass": False, "equal_FLOPs": False,
           "checkpoint_semantics": "cp0 after matched root accepted; cp5/10/15/20 after fresh frame encoding",
           "slots": [None] * 4, "pairs": [], "end_resource_settlement": None,
           "task_checkpoints": summary["pairs"],
           "limitations": ["source-trusted strict saved replay, not hardware attestation",
                           "control-step checkpoints, not same elapsed time",
                           "metered work and shapes do not establish equal FLOPs or total compute",
                           "known roots; no prespecified new control arm or fresh generalization"]}
    if receipt["full_saved_strict"] is not True:
        out["status"] = "partial_unmatched_unknown"
        out["pairs"] = [compare(None, None), compare(None, None)]
        return out
    report = budget.read_pinned(directory / "report.json", receipt["report_sha256"])
    out["end_resource_settlement"] = binding.settle_full_run({"run_path": str(directory)}, report)
    events = binding.host.storage.inspect_journal(directory / "events.jsonl", report["journal_sha256"])["events"]
    for row in report["rows"]:
        i = row["slot"]
        wrapped, positions = saved._events_with_positions(events, i)
        out["slots"][i] = checkpoint_costs(events, wrapped, positions, i)
        end = out["slots"][i]["20"]
        if end["attempted"] != budget.EXPECTED or end["completed"] != budget.EXPECTED:
            raise ValueError("checkpoint versus final numerical accounting")
    out["pairs"] = [compare(out["slots"][0], out["slots"][1]), compare(out["slots"][2], out["slots"][3])]
    for root, pair in enumerate(out["pairs"]):
        for step in CHECKPOINTS:
            pair[str(step)]["task"] = summary["pairs"][root]["checkpoints"][str(step)]
    out["status"] = "saved_checkpoint_work_descriptive"
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("run", "receipt", "receipt-sha", "prepared", "contract", "contract-sha", "output"):
        p.add_argument("--" + k, required=True)
    for k in ("plan", "plan-sha", "reference"):
        p.add_argument("--" + k)
    a = p.parse_args()
    output = Path(a.output)
    if not output.name.startswith(NAME + "_"):
        raise ValueError("exclusive own output prefix")
    print("Saved-only checkpoint work comparison; no GPU, model or simulator.", flush=True)
    result = analyze(a.run, a.receipt, a.receipt_sha, a.prepared, a.contract, a.contract_sha,
                     a.plan, a.plan_sha, a.reference)
    print(json.dumps({"sha256": budget.save_new(output, result), "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
