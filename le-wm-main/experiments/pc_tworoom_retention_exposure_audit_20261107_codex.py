"""Compare retention only before lookahead/effect execution histories diverge."""
import json
from collections import Counter

from experiments import pc_tworoom_temporal_new_episodes_20261107_codex as run


def compare(rows):
    counts = Counter()
    dispatch_divergence = []
    for row in rows:
        arms = row["arms"]
        old_a = old_b = None
        for a, b in zip(arms["lookahead"]["decisions"], arms["effect"]["decisions"]):
            if a["iteration"] != b["iteration"]:
                raise ValueError("unaligned decision clock")
            counts["same_dispatched_history"] += 1
            if old_a == old_b:
                counts["same_history_and_old_tail"] += 1
                if a["candidate_seed"] != b["candidate_seed"]:
                    raise ValueError("different candidate seeds")
                counts["different_retained"] += a["retained"] != b["retained"]
                counts["different_full_plan"] += a["actions"] != b["actions"]
                if a["dispatched"] != b["dispatched"]:
                    counts["different_dispatch"] += 1
                    dispatch_divergence.append([row["episode"], a["iteration"]])
            old_a = a["actions"][len(a["dispatched"]):]
            old_b = b["actions"][len(b["dispatched"]):]
            if a["dispatched"] != b["dispatched"]:
                break
    return dict(counts), dispatch_divergence


if __name__ == "__main__":
    source = json.loads(run.OUTPUT.read_text())
    counts, events = compare(source["rows"])
    print(json.dumps({"counts": counts, "first_divergences": events}, indent=2))
