"""Check common-tail intervention plan, decisions and recorded physical actions."""
import json

import numpy as np

from experiments import pc_shared_rgb_executed_chain_intervention_20261107_codex as exp
from experiments.pc_input_permissions import file_hash

OUT = exp.BASE.with_name(exp.BASE.name + "_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan, actual = json.loads(exp.PLAN.read_text()), json.loads(exp.PHYSICAL.read_text())
    if (actual["plan_sha256"] != file_hash(exp.PLAN) or plan["ledger_sha256"] != file_hash(exp.ledger.OUT)
            or plan["runner_sha256"] != file_hash(exp.__file__) or plan["protocol_sha256"] != file_hash(exp.PROTOCOL)):
        raise ValueError("source binding mismatch")
    original = json.loads(exp.ledger.OUT.read_text())
    summary = []
    for planned, observed, prior in zip(plan["rows"], actual["rows"], original["rows"], strict=True):
        root = planned["root"]
        if root != observed["root"] or root != prior["root"]:
            raise ValueError("root mismatch")
        previous = np.asarray([t["action"] for t in prior["trace"]], dtype=np.float32)
        prefixes = np.asarray(planned["prefix_candidates"], dtype=np.float32)
        np.testing.assert_array_equal(prefixes, exp.candidate_prefixes(previous[:5], 27123000 + root))
        np.testing.assert_array_equal(np.asarray(planned["common_tail"], dtype=np.float32), previous[5:])
        if planned["choices"] != {"original": 0, "zero": 1, "random": 2,
                                  "greedy": int(np.argmin(planned["greedy_scores"])),
                                  "lookahead": int(np.argmin(planned["lookahead_scores"]))}:
            raise ValueError("choice mismatch")
        for name in exp.ARMS:
            selected = np.concatenate((prefixes[planned["choices"][name]], previous[5:]))
            result = observed["outcomes"][name]
            trace = result["trace"]
            np.testing.assert_array_equal(np.asarray([x["action"] for x in trace], dtype=np.float32),
                                          selected[:len(trace)])
            if any(x["terminated"] or x["truncated"] for x in trace[:-1]):
                raise ValueError("execution continued after termination")
            if result["ever"] != any(x["terminated"] for x in trace) or \
                    result["prefix_success"] != any(x["terminated"] for x in trace[:5]):
                raise ValueError("success state mismatch")
        if observed["outcomes"]["original"]["trace"] != prior["trace"]:
            raise ValueError("original chain mismatch")
        summary.append({"root": root, "original": observed["outcomes"]["original"]["ever"],
                        "zero": observed["outcomes"]["zero"]["ever"]})
    audit = {"plan_sha256": file_hash(exp.PLAN), "physical_sha256": file_hash(exp.PHYSICAL),
             "runner_sha256": file_hash(__file__), "rows": summary,
             "scope": "sealed-score and trace audit; shared simulator, no model rescoring"}
    with OUT.open("x") as f:
        json.dump(audit, f, indent=2, allow_nan=False)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
