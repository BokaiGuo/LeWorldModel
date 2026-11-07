"""Check and summarize actual-chain interventions without upgrading mechanism claims."""
import json

import numpy as np

from experiments import pc_tworoom_actual_chain_intervention_20261107_codex as run
from experiments.pc_input_permissions import file_hash


def successes(arm):
    prefix = any(x["success"] for x in arm["prefix_trace"])
    return prefix, prefix or any(x["success"] for x in arm["common_tail_trace"]), \
        prefix or any(x["success"] for x in arm["replanned_trace"])


def main():
    result = json.loads(run.OUTPUT.read_text())
    if (result["source_sha256"] != file_hash(run.run.OUTPUT)
            or result["input_sha256"] != file_hash(run.run.INPUT)
            or result["protocol_sha256"] != file_hash(run.PROTOCOL)
            or result["runner_sha256"] != file_hash(run.__file__)):
        raise ValueError("artifact drift")
    eligible = [x for x in result["rows"] if "ineligible_reason" not in x]
    assert len(result["rows"]) == 40
    for row in eligible:
        for name in run.ARMS:
            arm = row["arms"][name]
            assert arm["queries"] == 16 and arm["packet_steps"] == 64
            assert len(arm["prefix_trace"]) == 5 and len(arm["replanned_actions"]) == 20
            assert np.abs(arm["prefix"]).max() <= 1 and np.abs(arm["replanned_actions"]).max() <= 1
            for trace in (arm["prefix_trace"], arm["common_tail_trace"], arm["replanned_trace"]):
                assert all(step["success"] == (step["distance"] < 16) for step in trace)
        assert successes(row["arms"]["actual"])[1] == row["original_success"]
    print("all", len(result["rows"]), "eligible", len(eligible),
          "ineligible", len(result["rows"]) - len(eligible))
    for name, subset in (
        ("eligible", eligible),
        ("eligible_non_initial", [r for r in eligible if not r["initial_success"]]),
        ("actual_prefix_not_success", [r for r in eligible if not r["initial_success"]
                                       and not successes(r["arms"]["actual"])[0]]),
    ):
        print("\n", name, len(subset))
        for arm in run.ARMS:
            vals = [successes(row["arms"][arm]) for row in subset]
            print(arm, "prefix/common/replanned", [sum(v[i] for v in vals) for i in range(3)])
        for other in ("lookahead", "zero", "random"):
            for column, label in ((1, "common"), (2, "replanned")):
                pairs = [(successes(row["arms"]["actual"])[column],
                          successes(row["arms"][other])[column]) for row in subset]
                print("actual vs", other, label, "win/loss",
                      sum(a and not b for a, b in pairs), sum(b and not a for a, b in pairs))
        if subset:
            a_less_progress = sum(
                row["arms"]["actual"]["prefix_trace"][-1]["distance"]
                >= row["arms"]["zero"]["prefix_trace"][-1]["distance"] for row in subset)
            print("actual prefix distance >= zero prefix distance", a_less_progress)


if __name__ == "__main__":
    main()
