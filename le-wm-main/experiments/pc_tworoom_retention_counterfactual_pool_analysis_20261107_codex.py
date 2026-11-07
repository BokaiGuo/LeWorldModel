"""Summarize physical success coverage without treating decisions as independent tasks."""
import json
from collections import Counter, defaultdict

import numpy as np

from experiments import pc_tworoom_retention_counterfactual_pool_20261107_codex as pool
from experiments.pc_input_permissions import file_hash


def classify(entry):
    prefix = entry["prefix"]
    ever = [any(s["success"] for s in trace) for trace in entry["physical"]]
    tail = [not entry["initial_success"] and not any(s["success"] for s in trace[:prefix])
            and any(s["success"] for s in trace[prefix:]) for trace in entry["physical"]]

    def covered(labels, name):
        return any(labels[2 * i + j] for i in entry["retained"][name] for j in range(2))

    return {name: {"pool": any(labels), "lookahead": covered(labels, "lookahead"),
                   "effect": covered(labels, "effect")}
            for name, labels in (("ever", ever), ("tail_after_no_prefix_success", tail))}


def main():
    data = json.loads(pool.OUTPUT.read_text())
    if (data["source_sha256"] != file_hash(pool.run.OUTPUT)
            or data["input_sha256"] != file_hash(pool.run.INPUT)
            or data["protocol_sha256"] != file_hash(pool.PROTOCOL)
            or data["runner_sha256"] != file_hash(pool.__file__)):
        raise ValueError("artifact drift")
    summary = {kind: Counter() for kind in ("ever", "tail_after_no_prefix_success")}
    per_task = {kind: defaultdict(lambda: [False, False, False]) for kind in summary}
    for entry in data["entries"]:
        chains = np.asarray(entry["actions"])
        assert chains.shape == (6, entry["remaining"], 2)
        assert np.isfinite(chains).all() and np.max(np.abs(chains)) <= 1
        assert len(entry["physical"]) == 6
        for trace in entry["physical"]:
            assert len(trace) <= entry["remaining"]
            assert all(t["success"] == (t["distance"] < 16) for t in trace)
        for kind, scores in classify(entry).items():
            for field in ("pool", "lookahead", "effect"):
                summary[kind][field] += scores[field]
            summary[kind]["effect_only"] += scores["effect"] and not scores["lookahead"]
            summary[kind]["lookahead_only"] += scores["lookahead"] and not scores["effect"]
            accumulated = per_task[kind][entry["episode"]]
            for i, field in enumerate(("pool", "lookahead", "effect")):
                accumulated[i] |= scores[field]
    print(f"matched decision pools: {len(data['entries'])}, episodes: {len({e['episode'] for e in data['entries']})}")
    for kind, counts in summary.items():
        roots = list(per_task[kind].values())
        print(kind, dict(counts))
        print("episode counts pool/lookahead/effect", [sum(r[i] for r in roots) for i in range(3)])


if __name__ == "__main__":
    main()
