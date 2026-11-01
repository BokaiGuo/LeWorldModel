"""Independently check saved first-feedback selection and exact node costs."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import p1_carry_saved_20261101_main as prior
from . import p1_first_feedback_carry_20261101_main as host

need, sha = prior.need, prior.sha


def check(folder):
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_bytes())
    need(report["status"] == "recorded_known_root_not_admitted", "completed source")
    results = []
    for row in report["rows"]:
        previous = None
        executed = np.asarray(row["executed_raw"], np.float32)
        for node in report["nodes"][row["episode_id"]]:
            if node["status"] == "not_started":
                continue
            need(node["status"] == "sealed", "sealed node")
            receipt, boundary = node["receipt"], node["boundary"]
            need(sha(receipt["trace"]) == receipt["trace_sha256"], "trace hash")
            events = [json.loads(x) for x in Path(receipt["trace"]).read_bytes().splitlines()]
            tail = None
            for index, event in enumerate(events):
                need(event["index"] == index and event["previous"] == tail and
                     event["sha256"] == prior.spec.digest({k: v for k, v in event.items() if k != "sha256"}),
                     "trace chain")
                tail = event["sha256"]

            def one(kind):
                found = [e["payload"] for e in events if e["kind"] == kind]
                need(len(found) == 1, kind + " multiplicity")
                return found[0]

            optimized = np.asarray(one("optimized_candidates")["raw"], np.float32)
            for event in events:
                if event["kind"] == "winner_refinement" and event["payload"]["accepted"]:
                    value = event["payload"]
                    optimized[value["index"]] = np.asarray(value["after_raw"], np.float32)
            plan = np.asarray(receipt["selected_full_plan"], np.float32)
            sealed = prior.binding.BoundPlan.seal(row["episode_id"], boundary, plan)
            need(sealed.plan_sha256 == receipt["selected_full_plan_sha256"]
                 and np.array_equal(plan, np.asarray(one("selected_full_plan")["raw"], np.float32)),
                 "full plan")
            horizon, has_carry = host.dimensions(boundary)
            extra = prior.binding.extra_cost(8, horizon, has_carry=has_carry) if boundary else {}
            source, scores = "initial", None
            if boundary:
                decision = one("first_feedback_selection")
                expected = optimized[:, :horizon]
                if has_carry:
                    expected = np.concatenate((expected, previous.plan[None, 1:]))
                need(np.array_equal(expected, np.asarray(decision["bank"], np.float32)), "candidate source")
                scores = decision["scores"]
                need(len(scores) == len(expected) and np.isfinite(scores).all(), "finite score roster")
                index = min(range(len(scores)), key=lambda i: (scores[i], i != 8 if has_carry else False, i))
                need(index == decision["selected_index"] == receipt["selected_index"]
                     and np.array_equal(plan, expected[index]), "selection and plan")
                source = "carry" if has_carry and index == 8 else "new"
                need(source == decision["selected_source"] and decision["incremental_cost"] == extra,
                     "source/cost label")
                cp = next(c for c in row["checkpoints"] if c["step"] == boundary)
                feedback = previous.bind_feedback(
                    episode=row["episode_id"], boundary=boundary,
                    issued_packet=executed[boundary-5:boundary], rgb_sha256=cp["pixel_sha256"])
                need({k: v for k, v in feedback.items() if k != "tail"} == decision["feedback"],
                     "feedback physical image binding")
            else:
                need(np.array_equal(plan, optimized[one("final_selection")["index"]]), "initial choice")
            commands = np.asarray(receipt["commands"], np.float32)
            actual = executed[boundary:boundary+5]
            need(np.array_equal(commands, plan[0]) and np.array_equal(actual, commands[:len(actual)]),
                 "physical action identity")
            ledger = receipt["ledger"]
            account = prior.consumer.consume(ledger["journal"], ledger["sha256"],
                                             receipt["contract_sha256"], row["arm"], "node", row["arm"])
            with patch.object(prior.control, "EXTRA_ARMS", ("structured", "native", "random")):
                cost = prior.control.cost(row["arm"])
            for key, value in extra.items():
                cost[key] += value
            need(account == receipt["account"] and account["charged_upper_bounds"] == cost
                 and receipt["module_hook_audit"]["ledger_charged_upper_bounds"] == cost,
                 "exact node ledger and model hooks")
            results.append(dict(arm=row["arm"], boundary=boundary, source=source, scores=scores))
            previous = sealed
    return dict(report_sha256=sha(folder / "report.json"), nodes=results, P1_admission=False,
                physical_event_chain_independently_checked=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.directory)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(nodes=len(result["nodes"]), output=str(args.output))))
