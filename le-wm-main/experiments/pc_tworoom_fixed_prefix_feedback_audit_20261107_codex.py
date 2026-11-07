"""Independent accounting and action handoff audit for fixed-prefix feedback."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_fixed_prefix_feedback_20261107_codex as run
from experiments.pc_input_permissions import file_hash

OUTPUT = Path(str(run.PREFIX) + "_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads(run.OUTPUT.read_text())
    if any(file_hash(path) != digest for path, digest in result["source_sha256"].items()):
        raise ValueError("source drift")
    checks = []
    with np.load(run.source.b.q.POOL, allow_pickle=False) as old:
        for row in result["rows"]:
            root = row["root"]
            chain = old[f"r{root}_actions"][0]
            open_arm, arm = row["arms"]["open"], row["arms"]["feedback"]
            run.check_same_prefix(chain, open_arm["trace"])
            run.check_same_prefix(chain, arm["trace"])
            trace = arm["trace"]
            previous = chain
            dispatched = 5
            for decision in arm["decisions"]:
                offset = 5 * decision["iteration"]
                tail = previous[dispatched:]
                chosen = np.asarray(decision["actions"], dtype=np.float32)
                scores = np.asarray(decision["all_scores"])
                if (decision["old_tail_sha256"] != run.digest(tail)
                        or decision["actions_sha256"] != run.digest(chosen)
                        or not decision["old_tail_included"]
                        or len(scores) != 680 or decision["queries"] != 680
                        or decision["winner"] != int(np.argmin(scores))
                        or decision["cost"] != float(scores[decision["winner"]])
                        or decision["packet_steps"] != 680 * decision["remaining"] // 5
                        or decision["forward_calls"] != 25):
                    raise ValueError("search accounting, old-tail or winner mismatch")
                count = min(5, len(trace) - offset)
                actual = np.asarray([step["action"] for step in trace[offset:offset + count]], dtype=np.float32)
                if not np.array_equal(actual, chosen[:count]):
                    raise ValueError("dispatched action mismatch")
                previous, dispatched = chosen, count
            if arm["queries"] != 680 * len(arm["decisions"]):
                raise ValueError("aggregate query mismatch")
            checks.append({"root": root, "decisions_verified": len(arm["decisions"]),
                           "success": arm["ever_success"], "queries": arm["queries"]})
    with OUTPUT.open("x") as f:
        json.dump({"source_sha256": file_hash(run.OUTPUT), "runner_sha256": file_hash(__file__),
                   "checks": checks, "scope": "stored data audit; no independent model rerun"}, f, indent=2)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
