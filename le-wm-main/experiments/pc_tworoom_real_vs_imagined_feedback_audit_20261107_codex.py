"""Verify stored actions, costs, matched budgets and previous RGB replay."""
import json
from pathlib import Path
import numpy as np

from experiments import pc_tworoom_real_vs_imagined_feedback_20261107_codex as run
from experiments.pc_input_permissions import file_hash

OUT = Path(str(run.PREFIX) + "_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    data = json.loads(run.OUTPUT.read_text())
    if any(file_hash(path) != value for path, value in data["source_sha256"].items()):
        raise ValueError("source drift")
    previous = json.loads(run.prior.OUTPUT.read_text())
    checks = []
    with np.load(run.prior.source.b.q.POOL, allow_pickle=False) as bank:
        for item in data["rows"]:
            root = item["root"]
            chain = bank[f"r{root}_actions"][0]
            actual = item["arms"]["real_rgb"]
            imagined = item["arms"]["imagined"]
            prior = next(r for r in previous["rows"] if r["root"] == root)["arms"]["feedback"]
            if actual["trace"] != prior["trace"]:
                raise ValueError("real arm differs from previous RGB feedback")
            for arm in (actual, imagined):
                run.prior.check_same_prefix(chain, arm["trace"])
                old, dispatched = chain, 5
                for decision in arm["decisions"]:
                    t = 5 * decision["iteration"]
                    chosen = np.asarray(decision["actions"], dtype=np.float32)
                    score = np.asarray(decision["all_scores"])
                    if (decision["old_tail_sha256"] != run.prior.digest(old[dispatched:]) or
                            decision["actions_sha256"] != run.prior.digest(chosen) or len(score) != 680 or
                            decision["winner"] != int(np.argmin(score)) or
                            decision["score"] != float(score[decision["winner"]]) or
                            decision["queries"] != 680 or decision["forward_batches"] != 25 or
                            decision["packet_steps"] != 680 * decision["remaining"] // 5):
                        raise ValueError("decision accounting mismatch")
                    dispatched = min(5, len(arm["trace"]) - t)
                    actual_actions = np.asarray([x["action"] for x in arm["trace"][t:t + dispatched]], dtype=np.float32)
                    if not np.array_equal(actual_actions, chosen[:dispatched]):
                        raise ValueError("dispatch mismatch")
                    old = chosen
            checks.append({"root": root, "rgb_prior_trace_exact": True,
                           "equal_queries": actual["queries"] == imagined["queries"],
                           "decision_action_equal": [a["actions_sha256"] == b["actions_sha256"]
                                                     for a, b in zip(actual["decisions"], imagined["decisions"])]})
    with OUT.open("x") as f:
        json.dump({"source_sha256": file_hash(run.OUTPUT), "runner_sha256": file_hash(__file__),
                   "checks": checks, "scope": "stored trace audit, not independent GPU rerun"}, f, indent=2)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
