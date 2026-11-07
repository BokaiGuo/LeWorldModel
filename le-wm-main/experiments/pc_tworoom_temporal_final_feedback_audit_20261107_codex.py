"""Audit sealed temporal feedback decisions without re-running the model."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_temporal_final_feedback_20261107_codex as run
from experiments.pc_input_permissions import file_hash
from experiments.pc_scored_candidate_selection_20261107_codex import ScoredCandidate, select_scored

OUT = run.ROOT / "results/pc_tworoom_temporal_final_feedback_audit_20261107_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    data = json.loads(run.OUT.read_text())
    original = json.loads(run.prior.OUT.read_text())
    if (data["runner_sha256"] != file_hash(run.__file__)
            or data["search_sha256"] != file_hash(run.search.__file__)
            or data["protocol_sha256"] != file_hash(run.PROTOCOL)
            or data["source_sha256"] != file_hash(run.prior.OUT)):
        raise ValueError("source drift")
    decisions, steps, identical = 0, 0, 0
    for row, old in zip(data["rows"], original["rows"], strict=True):
        if row["root"] != old["root"]:
            raise ValueError("root mismatch")
        for name, arm in row["arms"].items():
            strategy, rule = name.split("_")
            prior_tail = None
            cursor = 0
            for d in arm["decisions"]:
                count, remaining = len(d["dispatched"]), d["remaining_steps"]
                if remaining != 25 - cursor or d["model_queries"] != 16:
                    raise ValueError("incorrect clock or query count")
                expected_tail = None if prior_tail is None else run.prior.prior.action_hash(prior_tail)
                if d["old_tail_sha256"] != expected_tail:
                    raise ValueError("old-tail identity mismatch")
                actions = np.asarray(d["actions"], dtype=np.float32)
                dispatched = np.asarray(d["dispatched"], dtype=np.float32)
                if actions.shape != (remaining, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
                    raise ValueError("illegal actions")
                np.testing.assert_array_equal(actions[:count], dispatched)
                trace = arm["trace"][cursor:cursor + count]
                np.testing.assert_array_equal(dispatched, np.asarray([t["action"] for t in trace], dtype=np.float32))
                if any(bool(t["terminated"]) != (t["distance"] < 16.) or t["truncated"] for t in trace):
                    raise ValueError("physical success mismatch")
                prior_tail = actions[count:].copy()
                cursor += count
                if remaining == 5:
                    candidates = [ScoredCandidate("fallback", i, float(v)) for i, v in enumerate(d["parent_cost"])]
                else:
                    expected = run.search.base.choose_retained(
                        np.asarray(d["probe_cost"], dtype=np.float32), strategy, 27122000 + row["root"] * 10 + d["iteration"])
                    if expected != d["retained"]:
                        raise ValueError("retention mismatch")
                    vals = (d["parent_cost"], np.asarray(d["probe_cost"]).reshape(-1).tolist(), d["tail_cost"])
                    if rule == "temporal":
                        vals = d["temporal_scores"]
                    candidates = [ScoredCandidate(source, i, float(v))
                                  for source, values in zip(("parent", "probe", "tail"), vals, strict=True)
                                  for i, v in enumerate(values)]
                choice = select_scored(candidates)
                if (choice.source, choice.index) != (d["choice"]["source"], d["choice"]["index"]):
                    raise ValueError("choice mismatch")
                decisions += 1
                steps += count
            if cursor != len(arm["trace"]) or arm["ever_success"] != any(t["terminated"] for t in arm["trace"]):
                raise ValueError("episode trace mismatch")
            if rule == "endpoint":
                old_arm = old["arms"][strategy]
                if arm["trace"] != old_arm["trace"] or [d["actions_sha256"] for d in arm["decisions"]] != [
                    d["actions_sha256"] for d in old_arm["decisions"]
                ]:
                    raise ValueError("endpoint diverges from prior run")
                identical += 1
    with OUT.open("x") as stream:
        json.dump({"source_sha256": file_hash(run.OUT), "runner_sha256": file_hash(Path(__file__)),
                   "decisions_checked": decisions, "physical_steps_checked": steps,
                   "endpoint_episodes_identical": identical,
                   "scope": "record arithmetic, bindings and prior trace equality; not independent replay"},
                  stream, indent=2)
    print(f"verified {decisions} decisions, {steps} physical steps, {identical} endpoint traces")


if __name__ == "__main__":
    main()
