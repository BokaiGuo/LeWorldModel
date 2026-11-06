"""Independent audit of sealed-plan intervention logs and full-tail probes."""
import json

import numpy as np

from experiments import pc_retention_mechanism_next_dev as source


def main():
    out, seal = source.OUT, source.seal
    target = out / "interventions_audit.json"
    if target.exists():
        raise FileExistsError(target)
    manifest = json.loads((out / "manifest.json").read_text())
    plans = json.loads((out / "plans.json").read_text())
    data = json.loads((out / "interventions.json").read_text())
    seal.verify(manifest["dependencies"])
    if (data["plans_sha256"] != seal.digest(out / "plans.json") or
            data["plans_audit_sha256"] != seal.digest(out / "plans_audit.json") or len(data["rows"]) != 32):
        raise ValueError("unbound results")
    for plan, row in zip(plans["rows"], data["rows"], strict=True):
        if (row["root"], row["arm"]) != (plan["root"], plan["arm"]):
            raise ValueError("identity mismatch")
        if row["status"] != "completed":
            continue
        root, arm, task = row["root"], row["arm"], plan["task"]
        if json.loads((out / f"intervention_{root:02d}_{arm}.json").read_text()) != row:
            raise ValueError("root output changed")
        info = source.journal_audit.audit_journal(out / f"intervention_{root:02d}_{arm}_calls")
        if info["tokens"] != row["tokens"] or len(info["calls"]) != row["calls"]:
            raise ValueError("intervention ledger mismatch")
        calls = list(source.old.previous.fixed.read_calls(out / f"intervention_{root:02d}_{arm}_calls"))
        for actions, features, scores in calls:
            path, _ = source.old.OLD.previous.independent(task["start"], actions)
            np.testing.assert_allclose(path[:, -1, :2], features, atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, np.linalg.norm(features - task["goal"], axis=1), atol=1e-12, rtol=0)
        chain = np.asarray(plan["result"]["selected"], np.float32)
        np.testing.assert_array_equal(chain, np.asarray(row["selected_chain"], np.float32))
        events = [json.loads(p.read_text()) for p in sorted((out / f"intervention_{root:02d}_{arm}_events").glob("*.json"))]
        if len(events) != 30:
            raise ValueError("missing full-tail events")
        for i in range(0, len(events), 3):
            request, branch_request, result = events[i:i+3]
            if [request["event"], branch_request["event"], result["event"]] != ["request", "branch_request", "result"]:
                raise ValueError("event order mismatch")
            name, eps = request["control"], request["epsilon"]
            for item in (branch_request, result):
                if any(item[k] != request[k] for k in ("control", "epsilon", "chain_sha256_float32")):
                    raise ValueError("event binding mismatch")
            prefix = row["suffix"][name]["prefix"]
            expected = source.response.prepare(chain, prefix, epsilon=eps)
            np.testing.assert_array_equal(np.asarray(request["actions"], np.float32), expected["actions"])
            if request["specs"] != expected["specs"] or request["chain_sha256_float32"] != expected["selected_chain_sha256_float32"]:
                raise ValueError("probe specification mismatch")
            actions = expected["actions"]
            matching = [(features, scores) for recorded, features, scores in calls
                        if recorded.shape == actions.shape and np.array_equal(recorded, actions)]
            if not matching:
                raise ValueError("probe missing from prediction journal")
            path, blocked = source.old.OLD.previous.independent(task["start"], actions)
            value = result["result"]
            if not any(np.array_equal(features, np.asarray(value["endpoint_positions"])) and
                       np.array_equal(scores, np.asarray(value["scores"])) for features, scores in matching):
                raise ValueError("probe output not bound to prediction journal")
            np.testing.assert_allclose(path[:, -1, :2], value["endpoint_positions"], atol=1e-12, rtol=0)
            # Independent geometry's branch encoding may differ; recheck the
            # exact branch encoding with the implementation used by the run.
            _, actual_blocked = source.exact.wall.rollout(task["start"],
                np.pad(actions, ((0, 0), (0, 1), (0, 0))) * source.exact.wall.HIGH)
            np.testing.assert_array_equal(np.asarray(actual_blocked)[:, :15], value["branches"])
            rebuilt = source.response.evaluate(expected, np.asarray(value["endpoint_positions"]),
                                                np.asarray(value["branches"]), task["goal"])
            for key in rebuilt:
                if rebuilt[key] != value[key]:
                    raise ValueError("response reconstruction mismatch")
        print(f"audited intervention root={root} arm={arm}", flush=True)
    seal.verify(manifest["dependencies"])
    source.journal.write_json(target, {"status": "passed", "interventions_sha256": seal.digest(out / "interventions.json"),
        "scope": "same environment geometry and response reconstruction; not independent replication"})


if __name__ == "__main__":
    main()
