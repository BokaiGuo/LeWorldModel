"""Independently recompute fixed-pool retention and success coverage."""
import json

import numpy as np

from experiments import pc_tworoom_temporal_retention_fixed_pool_20261107_codex as run
from experiments.pc_input_permissions import file_hash


def main():
    plan, result, physical = (json.loads(p.read_text()) for p in (run.PLAN, run.OUT, run.pool.OUTPUT))
    if (result["plan_sha256"] != file_hash(run.PLAN) or result["pool_sha256"] != file_hash(run.pool.OUTPUT)
            or result["protocol_sha256"] != file_hash(run.PROTOCOL)
            or result["runner_sha256"] != file_hash(run.__file__)):
        raise ValueError("artifact drift")
    entries = {(r["episode"], r["iteration"]): r for r in physical["entries"]}
    output = {(r["episode"], r["iteration"]): r for r in result["rows"]}
    different = 0
    for record in plan["decisions"]:
        key = (record["episode"], record["iteration"])
        entry, observed = entries[key], output[key]
        for name, scores, chosen in (("baseline", record["endpoint"], record["baseline"]),
                                     ("temporal", record["temporal"], record["temporal_kept"])):
            values = np.asarray(scores)
            expected = sorted(range(3), key=lambda i: (float(min(values[i])), i))[:2]
            if chosen != expected:
                raise ValueError("selection mismatch")
            prefixes = set(chosen)
            ever = False
            tail = False
            for j, trace in enumerate(entry["physical"]):
                if j // 2 not in prefixes:
                    continue
                successes = [x["success"] for x in trace]
                ever |= any(successes)
                tail |= (not entry["initial_success"] and not any(successes[:entry["prefix"]])
                         and any(successes[entry["prefix"]:]))
            if observed[name] != {"ever": bool(ever), "tail": bool(tail)}:
                raise ValueError("coverage mismatch")
        different += set(record["baseline"]) != set(record["temporal_kept"])
    if len(plan["decisions"]) != 119 or len(output) != 119 or len(entries) != 119:
        raise ValueError("panel mismatch")
    print("audited", len(output), "retention_set_changes", different)


if __name__ == "__main__":
    main()
