"""Audit sealed closed-loop trace, score accounting, action identity and clock."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments import pc_rgb_block_incumbent_search_20261107_codex as planner
from experiments import pc_tworoom_incumbent_rgb_feedback_20261107_codex as run

OUT = run.ROOT / "results/pc_tworoom_incumbent_rgb_feedback_audit_20261107_codex.json"


def audit(record):
    if record["track"] != "primary_rgb" or [r["root"] for r in record["rows"]] != list(range(10, 14)):
        raise ValueError("track or roots")
    summary = {arm: {"successes": 0, "decisions": 0, "queries": 0, "packet_steps": 0,
                     "forward_calls": 0, "planning_seconds": 0.0, "env_steps": 0,
                     "old_tail_wins": 0, "source_wins": 0, "new_wins": 0} for arm in planner.ARMS}
    for root in record["rows"]:
        if set(root["arms"]) != set(planner.ARMS):
            raise ValueError("arm set")
        for arm, episode in root["arms"].items():
            trace = episode["trace"]
            cursor = 0
            old_tail = None
            for iteration, decision in enumerate(episode["decisions"]):
                remaining = 25 - cursor
                if decision["iteration"] != iteration or decision["remaining"] != remaining:
                    raise ValueError("decision clock")
                actions = np.asarray(decision["actions"], dtype=np.float32)
                digest = hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()
                if (actions.shape != (remaining, 2) or not np.isfinite(actions).all()
                        or np.any(np.abs(actions) > 1) or digest != decision["actions_sha256"]):
                    raise ValueError("actions or fingerprint")
                if decision["old_tail_included"] != (old_tail is not None):
                    raise ValueError("old tail flag")
                if old_tail is not None and decision["winner"] == 0:
                    if not np.array_equal(actions, old_tail):
                        raise ValueError("old tail identity")
                    summary[arm]["old_tail_wins"] += 1
                scores = np.asarray(decision["all_scores"])
                if (scores.shape != (680,) or not np.isfinite(scores).all()
                        or decision["winner"] != int(np.argmin(scores))
                        or decision["score"] != float(scores[decision["winner"]])):
                    raise ValueError("selection")
                expected_batches = ([100, 100, 100, 300, 80] if arm == "baseline" or remaining == 5
                                    else [100, 100, 100, 300, 64, 16])
                if (decision["query_batches"] != expected_batches or decision["queries"] != 680
                        or decision["packet_steps"] != 680 * (remaining // 5)):
                    raise ValueError("query accounting")
                receipts = decision["receipts"]
                expected_calls = sum((n + 31) // 32 for n in expected_batches)
                if len(receipts) != expected_calls or decision["forward_calls"] != expected_calls:
                    raise ValueError("forward accounting")
                if any(r["track_id"] != "primary_rgb" or r["operation"] != "score" for r in receipts):
                    raise ValueError("receipt track")
                if sum(r["inputs"]["candidate_actions"]["shape"][1] for r in receipts) != 680:
                    raise ValueError("receipt candidates")
                n = decision["dispatched"]
                if n not in range(1, 6) or cursor + n > len(trace):
                    raise ValueError("dispatch count")
                for j in range(n):
                    if not np.array_equal(np.asarray(trace[cursor+j]["action"], np.float32), actions[j]):
                        raise ValueError("dispatched action mismatch")
                    if j < n - 1 and (trace[cursor+j]["terminated"] or trace[cursor+j]["truncated"]):
                        raise ValueError("actions executed after termination")
                cursor += n
                if cursor < len(trace) and (trace[cursor-1]["terminated"] or trace[cursor-1]["truncated"]):
                    raise ValueError("replanned after terminal")
                old_tail = actions[n:].copy()
                s = summary[arm]
                s["decisions"] += 1
                s["queries"] += 680
                s["packet_steps"] += decision["packet_steps"]
                s["forward_calls"] += expected_calls
                s["planning_seconds"] += decision["planning_seconds"]
                s["source_wins"] += decision["winner"] < 600
                s["new_wins"] += decision["winner"] >= 600
            if cursor != len(trace) or len(trace) > 25 or episode["queries"] != 680 * len(episode["decisions"]):
                raise ValueError("episode accounting")
            if episode["ever_success"] != any(t["terminated"] for t in trace):
                raise ValueError("success accounting")
            summary[arm]["successes"] += episode["ever_success"]
            summary[arm]["env_steps"] += len(trace)
    return summary


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    record = json.loads(run.OUT.read_text())
    for key, path in (("protocol_sha256", run.PROTOCOL), ("runner_sha256", run.__file__),
                      ("planner_sha256", planner.__file__)):
        if record[key] != file_hash(Path(path)):
            raise ValueError(f"binding mismatch: {key}")
    summary = audit(record)
    with OUT.open("x") as stream:
        json.dump({"input_sha256": file_hash(run.OUT), "auditor_sha256": file_hash(__file__),
                   "summary": summary, "scope": "trace audit, not independent scientific confirmation"},
                  stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
