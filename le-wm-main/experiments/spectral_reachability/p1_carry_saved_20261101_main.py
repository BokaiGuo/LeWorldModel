"""Independent saved-plan, selection, command and ledger checks for carry runs."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import codex_four_arm_full_cost_20261030 as spec
from . import codex_four_arm_consumer_20261030 as consumer
from . import p1_carry_binding_20261101 as binding
from . import p1_carry_host_20261101_main as host
from . import p1_four_arm_extra32_budget_20261031 as control


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


def check(folder, expected_sha):
    folder = Path(folder)
    report_path = folder / "report.json"
    need(sha(report_path) == expected_sha, "report hash")
    report = json.loads(report_path.read_bytes())
    need(report["status"] == "recorded_known_root_not_admitted", "incomplete run")
    results = []
    for row in report["rows"]:
        previous = None
        executed = np.asarray(row["executed_raw"], np.float32)
        for node in report["nodes"][row["episode_id"]]:
            if node["status"] == "not_started":
                continue
            need(node["status"] == "sealed", "failed node")
            receipt = node["receipt"]
            boundary = node["boundary"]
            need(sha(receipt["trace"]) == receipt["trace_sha256"], "trace hash")
            events = [json.loads(x) for x in Path(receipt["trace"]).read_bytes().splitlines()]
            tail = None
            for i, event in enumerate(events):
                unhashed = {k: v for k, v in event.items() if k != "sha256"}
                need(event["index"] == i and event["previous"] == tail
                     and event["sha256"] == spec.digest(unhashed), "trace chain")
                tail = event["sha256"]
            def one(kind):
                matches = [e["payload"] for e in events if e["kind"] == kind]
                need(len(matches) == 1, "missing or repeated " + kind)
                return matches[0]
            plan = np.asarray(receipt["selected_full_plan"], np.float32)
            sealed = binding.BoundPlan.seal(row["episode_id"], boundary, plan)
            need(sealed.plan_sha256 == receipt["selected_full_plan_sha256"]
                 and np.array_equal(plan, np.asarray(one("selected_full_plan")["raw"], np.float32)),
                 "full plan receipt")
            optimized = np.asarray(one("optimized_candidates")["raw"], np.float32)
            refinements = [e["payload"] for e in events if e["kind"] == "winner_refinement"]
            if refinements and refinements[0]["accepted"]:
                optimized[refinements[0]["index"]] = np.asarray(refinements[0]["after_raw"], np.float32)
            source = "original"
            if boundary >= 10:
                carry = one("carry_selection")
                horizon = (20 - boundary) // 5
                expected_bank = np.concatenate((optimized[:, :horizon], previous.plan[None, 1:]))
                need(np.array_equal(expected_bank, np.asarray(carry["candidate_plans"], np.float32)),
                     "carry candidate source")
                scores = carry["scores"]
                need(len(scores) == 9 and np.isfinite(scores).all(), "finite score denominator")
                index = min(range(9), key=lambda i: (scores[i], i != 8, i))
                need(index == receipt["selected_index"] == carry["selected_index"]
                     and np.array_equal(plan, expected_bank[index]), "model-only selection")
                need(carry["incremental_cost"] == host.node_extra(boundary), "extra cost")
                feedback = previous.bind_feedback(
                    episode=row["episode_id"], boundary=boundary,
                    issued_packet=executed[boundary-5:boundary],
                    rgb_sha256=carry["feedback"]["rgb_sha256"])
                need({k: v for k, v in feedback.items() if k != "tail"} ==
                     carry["feedback"] == receipt["carry_feedback"], "feedback binding")
                source = carry["selected_source"]
                need(source == ("carry" if index == 8 else "new"), "source label")
            else:
                need(np.array_equal(plan, optimized[one("final_selection")["index"]]),
                     "original model selection")
            commands = np.asarray(receipt["commands"], np.float32)
            need(np.array_equal(commands, plan[0]), "packet extraction")
            actual = executed[boundary:boundary+5]
            need(np.array_equal(actual, commands[:len(actual)]), "actual command prefix")
            ledger = receipt["ledger"]
            account = consumer.consume(ledger["journal"], ledger["sha256"],
                                       receipt["contract_sha256"], row["arm"], "node", row["arm"])
            with patch.object(control, "EXTRA_ARMS", ("structured", "native", "random")):
                expected_cost = control.cost(row["arm"])
            for key, value in host.node_extra(boundary).items():
                expected_cost[key] += value
            need(account == receipt["account"] and account["charged_upper_bounds"] == expected_cost,
                 "exact saved accounting")
            need(receipt["module_hook_audit"]["ledger_charged_upper_bounds"] == expected_cost,
                 "hook account")
            results.append(dict(episode=row["episode_id"], boundary=boundary,
                                selected_source=source, executed_steps=len(actual)))
            previous = sealed
    return dict(report_sha256=expected_sha, nodes=results,
                physical_event_chain_independently_checked=False,
                P1_admission=False, independent_confirmation=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = check(args.directory, args.report_sha)
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"checked_nodes": len(result["nodes"]), "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
