"""Compare saved AP development host journals with their exposed reference run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import pc_ap_durable_saved_consumer_20261102_codexexclusive as base


def compare_slot(row, events):
    """Exact comparison of fields retained by both producers; no physical replay."""
    require = base.require
    seed, arm = row["seed"], row["arm"]
    require(row["status"] == "complete", "reference arm incomplete")
    warm = next(e for e in events if e["kind"] == "warmup_intent")
    require((warm["seed"], warm["arm"], warm["history_index"])
            == (seed, arm, row["history_index"]), "warmup seed/arm/history index")
    starts = [e for e in events if e["kind"] == "planning_intent"]
    plans = [e for e in events if e["kind"] == "planning_returned"]
    bound = [e for e in events if e["kind"] == "context_bound"]
    committed = [e for e in events if e["kind"] == "context_committed"]
    distances = [e for e in events if e["kind"] == "evaluator_distance"]
    steps = [e["event"] for e in events if e["kind"] == "bridge_event"
             and e["event"]["kind"] == "step_returned"]
    require(len(starts) == len(plans) == len(bound) == len(committed) == len(row["decisions"]) == 2,
            "decision count")
    require(len(distances) == len(steps) == len(row["steps"]) == 10, "step count")
    for index, reference in enumerate(row["decisions"]):
        actual = plans[index]["audit"]
        require(starts[index]["seed"] == actual["search_seed"] == reference["search_seed"]
                == seed + 8100 + index, f"decision {index} search seed")
        for field in ("raw_actions", "normalized_actions"):
            x = np.asarray(actual[field], dtype="<f4")
            y = np.asarray(reference[field], dtype="<f4")
            require(x.shape == y.shape == (5, 2) and x.tobytes() == y.tobytes(),
                    f"decision {index} {field} bytes")
        require(committed[index]["receipt"]["planned_action_sha256"] == reference["advanced_action_sha256"],
                f"decision {index} action sha")
        require(bound[index]["context"]["root"]["sha256"] == reference["root_rgb_sha256"]
                and committed[index]["context"]["root"]["sha256"] == reference["successor_rgb_sha256"],
                f"decision {index} RGB sha")
        for field in ("prediction_samples", "prediction_batches"):
            require(actual[field] == reference[field], f"decision {index} {field}")
    require(bound[0]["context"]["goal"]["sha256"] == row["goal_rgb_sha256"]
            and bound[0]["context"]["root"]["sha256"] == row["root_rgb_sha256"], "root/goal RGB")
    for index, (reference, distance, actual) in enumerate(zip(row["steps"], distances, steps), 1):
        require(reference["distance"] == distance["qpos_distance"], f"step {index} distance")
        for field in ("reward", "terminated", "truncated"):
            require(reference[field] == actual[field], f"step {index} {field}")
    return {"status": "matched", "seed": seed, "arm": arm, "history_index": warm["history_index"],
            "decisions": 2, "search_seeds": [e["seed"] for e in starts],
            "raw_action_float32_values": 20, "normalized_action_float32_values": 20,
            "physical_steps": 10, "final_distance": distances[-1]["qpos_distance"],
            "matched_rgb": ["initial_root", "goal", "two_decision_roots", "two_successors"]}


def audit(directory, reference_path):
    result = {"scope": "replay of four exposed development slots, not new independent outcomes",
              "slots": {}, "errors": []}
    integrity = base.audit(directory, reference_path)
    result["saved_consistency_status"] = integrity["status"]
    result["saved_consistency_errors"] = integrity["errors"]
    if integrity["errors"]:
        result["errors"].append("saved journal consistency failed")
    try:
        raw = reference_path.read_bytes()
        base.require(base.sha(raw) == base.REF_SHA, "reference SHA mismatch")
        reference = json.loads(raw)
        result["reference_sha256"] = base.sha(raw)
        for seed in base.SEEDS:
            for arm in base.ARMS:
                name = f"{seed}_{arm}"
                try:
                    rows = [r for r in reference["rows"] if r["seed"] == seed and r["arm"] == arm]
                    base.require(len(rows) == 1, "reference slot cardinality")
                    events, chain = base.read_chain(directory / f"{name}.jsonl")
                    result["slots"][name] = compare_slot(rows[0], events)
                    result["slots"][name]["journal_sha256"] = chain["file_sha256"]
                except Exception as exc:
                    result["slots"][name] = {"status": "unmatched_or_incomplete",
                                             "error": f"{type(exc).__name__}: {exc}"}
                    result["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
    except Exception as exc:
        result["errors"].append(f"reference: {type(exc).__name__}: {exc}")
    result["status"] = "reference_development_replay_match" if not result["errors"] else "unmatched_or_incomplete"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=base.INPUT)
    parser.add_argument("--reference", type=Path, default=base.REFERENCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input, args.reference)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps({"status": result["status"], "errors": result["errors"]}))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
