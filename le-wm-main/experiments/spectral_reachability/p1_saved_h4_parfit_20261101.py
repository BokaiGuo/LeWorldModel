"""Read-only saved H4 result checker; no GPU, model or physical execution."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_four_arm_consumer_20261030 as consumer
from . import codex_four_arm_full_cost_20261030 as spec
from . import p1_carry_binding_20261101 as binding
from . import p1_h4_parfit_20261101 as h4


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chain(path, physical=False):
    events = []
    previous = None
    for index, line in enumerate(Path(path).read_bytes().splitlines()):
        event = json.loads(line)
        key, prev = ("event_sha256", "previous_sha256") if physical else ("sha256", "previous")
        body = {k: v for k, v in event.items() if k != key}
        computed = (hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
                    if physical else spec.digest(body))
        need(event["index"] == index and event[prev] == previous and event[key] == computed,
             "journal hash chain")
        previous = event[key]
        events.append(event)
    need(bool(events), "empty journal")
    return events


def status_only(directory, report_sha):
    """Classify the recorded branch, without treating a partial run as verified."""
    report_path = Path(directory).resolve() / "report.json"
    need(spec.is_hash(report_sha) and sha(report_path) == report_sha, "external report hash")
    report = json.loads(report_path.read_bytes())
    need(len(report["rows"]) == 1, "single-arm report required")
    row = report["rows"][0]
    count = row["control_returned"]
    need(type(count) is int and 0 <= count <= 20, "invalid returned step count")
    if count == 20 and row.get("full_horizon") is True:
        classification = "full20_eligible_for_strict_check"
    elif row.get("terminated") is True or row.get("truncated") is True:
        classification = "native_early_termination_or_truncation"
    else:
        classification = "partial_or_error_requires_inspection"
    return {"status": "report_only_not_integrity_verified", "classification": classification,
            "arm": row["arm"], "control_returned": count,
            "native_success": row.get("native_success"), "row_status": row.get("status"),
            "terminated": row.get("terminated"), "truncated": row.get("truncated"),
            "report_status": report.get("status"), "errors": report.get("errors"),
            "sealed_boundaries": [node["boundary"] for node in report["nodes"][row["episode_id"]]
                                  if node["status"] == "sealed"],
            "report_sha256": report_sha, "P1_admission": False,
            "limitations": "classification only; sources, ledgers, actions and journals not checked"}


def check(directory, report_sha):
    directory = Path(directory).resolve()
    report_path = directory / "report.json"
    need(spec.is_hash(report_sha) and sha(report_path) == report_sha, "external report hash")
    report = json.loads(report_path.read_bytes())
    need(report["status"] == "recorded_known_root_not_admitted"
         and report["P1_admission"] is False and len(report["rows"]) == 1, "completed H4 report")
    prepared = Path(report["prepared"])
    need(sha(prepared) == report["prepare_sha256"], "prepared hash")
    material = json.loads(prepared.read_bytes())
    need(material["schema"] == "p1_reverse_h4_parfit_20261101.prepare.v1", "H4 prepare schema")
    need(report["protocol"] == material["protocol"], "prepared/report protocol mismatch")
    for path, expected in material["sources"].items():
        need(sha(path) == expected, "source hash: " + path)
    row = report["rows"][0]
    arm, episode = row["arm"], row["episode_id"]
    need(arm in h4.ARMS and row["control_returned"] == 20 and row["full_horizon"] is True,
         "completed physical episode")
    physical = chain(directory / "journal.jsonl", physical=True)
    need(sha(directory / "journal.jsonl") == report["journal_sha256"]
         and physical[-1]["event_sha256"] == report["journal_chain_tail"],
         "physical journal report binding")
    returns = [e for e in physical if e["kind"] == "control_return" and e["episode_id"] == episode]
    dispatches = [e for e in physical if e["kind"] == "authorized_dispatch" and e["episode_id"] == episode]
    need([e["step"] for e in returns] == list(range(1, 21))
         and [e["step"] for e in dispatches] == list(range(1, 21)), "physical return sequence")
    executed = np.asarray(row["executed_raw"], np.float32)
    need(executed.shape == (20, 2), "returned action shape")
    for i, (issued, returned) in enumerate(zip(dispatches, returns)):
        need(issued["index"] < returned["index"]
             and np.array_equal(np.asarray(issued["command"], np.float32), executed[i])
             and np.array_equal(np.asarray(returned["command"], np.float32), executed[i]),
             "issued/returned action mismatch")
    nodes = report["nodes"][episode]
    need([n["boundary"] for n in nodes] == [0, 5, 10, 15], "four ordered nodes")
    sealed_events = [e for e in physical if e["kind"] == "boundary_sealed"
                     and e["episode_id"] == episode]
    need([e["boundary"] for e in sealed_events] == [0, 5, 10, 15], "physical boundary seals")
    previous = None
    results = []
    for entry in nodes:
        boundary, receipt = entry["boundary"], entry["receipt"]
        horizon, carry = h4.dimensions(boundary)
        need(entry["status"] == "sealed" and receipt["status"] == "complete", "node complete")
        seal_event = sealed_events[boundary // 5]
        receipt_digest = spec.digest(receipt)
        need(seal_event["plan_sha256"] == receipt_digest
             and seal_event["index"] < dispatches[boundary]["index"]
             and (not boundary or returns[boundary - 1]["index"] < seal_event["index"]),
             "physical boundary seal/order")
        trace = Path(receipt["trace"])
        need(trace.resolve().is_relative_to(directory) and sha(trace) == receipt["trace_sha256"],
             "node trace hash/path")
        events = chain(trace)
        def one(kind):
            matches = [e["payload"] for e in events if e["kind"] == kind]
            need(len(matches) == 1, "event count: " + kind)
            return matches[0]
        config = one("h4_contract_binding")
        need(config["boundary"] == boundary
             and config["config"]["planning_horizon_packets"] == horizon
             and config["config"]["allocated_horizon_packets"] == 4
             and config["config"]["decision_boundary"] == boundary
             and config["policy"]["horizon"] == horizon
             and spec.digest(config["config"]) == config["config_sha256"]
             and spec.digest(config["policy"]) == config["policy_sha256"]
             and config["contract_sha256"] == receipt["contract_sha256"]
             and config["projection"] == "identity_192_exact", "H4 contract binding")
        ledger = receipt["ledger"]
        need(Path(ledger["journal"]).resolve().is_relative_to(directory), "ledger path")
        first = json.loads(Path(ledger["journal"]).read_bytes().splitlines()[0])["payload"]["contract"]
        need(spec.digest(first) == receipt["contract_sha256"]
             and first["identity"]["optimizer_config"] == config["config_sha256"]
             and first["phase_policy_sha256"] == config["policy_sha256"], "durable contract hashes")
        account = consumer.consume(ledger["journal"], ledger["sha256"],
                                   receipt["contract_sha256"], arm, "node", arm)
        cost = h4.node_cost(arm, boundary)
        audit = receipt["module_hook_audit"]
        need(account == receipt["account"] and account["status"] == "closed_unverified"
             and account["charged_upper_bounds"] == cost
             and audit["ledger_charged_upper_bounds"] == cost
             and audit["module_forward_pre_hooks"]["predict_calls"] == cost["predict_calls"]
             and audit["module_forward_pre_hooks"]["action_encode_calls"] == cost["action_encode_calls"]
             and audit["autograd_api_counts"]["grad_attempted"]
             == audit["autograd_api_counts"]["grad_completed"] == cost["backward_calls"],
             "node charged ledger/hook mismatch")
        event = one("selected_full_plan")
        plan = np.asarray(receipt["selected_full_plan"], np.float32)
        sealed = binding.BoundPlan.seal(episode, boundary, plan)
        need(plan.shape == (horizon, 5, 2) and sealed.plan_sha256 == receipt["selected_full_plan_sha256"]
             and np.array_equal(plan, np.asarray(event["raw"], np.float32))
             and event["charges"] == cost and event["horizon"] == horizon
             and event["index"] == receipt["selected_index"], "selected plan receipt")
        bank = np.asarray(one("optimized_candidates_before_exchange")["raw"], np.float32)
        need(bank.shape == (8, 4, 5, 2) and np.isfinite(bank).all(), "fresh bank")
        if horizon > 1:
            exchange = one("suffix_exchange")
            scores = exchange["scores"]
            need(len(scores) == 64 and np.isfinite(scores).all(), "exchange score shape")
            tails = [min(range(8), key=lambda j: (scores[8*i+j], j != i, j)) for i in range(8)]
            need(tails == exchange["tail_indices"], "exchange argmin")
            bank = np.stack([np.concatenate((bank[i, :1], bank[j, 1:])) for i, j in enumerate(tails)])
            if not carry:
                need(event["scores"] == [scores[8*i+j] for i, j in enumerate(tails)],
                     "initial exchange score binding")
        selection_bank = bank[:, :horizon]
        feedback = one("h4_feedback_binding")
        cp = next(x for x in row["checkpoints"] if x["step"] == boundary)
        need(feedback["boundary"] == boundary and feedback["horizon"] == horizon
             and feedback["current_rgb_sha256"] == cp["pixel_sha256"], "current RGB identity")
        if carry:
            linked = previous.bind_feedback(episode=episode, boundary=boundary,
                issued_packet=executed[boundary-5:boundary], rgb_sha256=cp["pixel_sha256"])
            need(feedback["feedback"] == {k: v for k, v in linked.items() if k != "tail"},
                 "previous plan, returned packet and current RGB binding")
            selection_bank = np.concatenate((selection_bank, linked["tail"][None]))
        else:
            need(feedback["feedback"] is None, "unexpected first feedback")
        scores = event["scores"]
        need(len(scores) == len(selection_bank) and np.isfinite(scores).all()
             and scores == receipt["scores"], "selection scores")
        index = min(range(len(scores)), key=lambda i: (scores[i], i != 8 if carry else False, i))
        need(index == event["index"] and np.array_equal(plan, selection_bank[index]),
             "argmin or carried-tail identity")
        need(np.array_equal(plan[0], np.asarray(receipt["commands"], np.float32))
             and np.array_equal(plan[0], executed[boundary:boundary+5]), "actual packet")
        results.append({"boundary": boundary, "horizon": horizon, "selected_index": index,
                        "source": "carry" if carry and index == 8 else "fresh", "charges": cost})
        previous = sealed
    return {"status": "saved_integrity_checked_not_scientific_admission", "report_sha256": report_sha,
            "source_count": len(material["sources"]), "physical_event_hash_chain_checked": True,
            "nodes": results, "P1_admission": False, "independent_confirmation": False,
            "limitations": "no independent GPU kernel accounting, score recomputation or simulator replay"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--status-only", action="store_true",
                        help="classify partial/native-stop outcomes without integrity admission")
    args = parser.parse_args()
    result = status_only(args.directory, args.report_sha256) if args.status_only else check(
        args.directory, args.report_sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
