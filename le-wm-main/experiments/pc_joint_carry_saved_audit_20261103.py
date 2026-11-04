"""Offline binding audit of the exposed-root joint-plan carry experiment."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/pc_causal_joint_carry_tail_20261103_DEV01.json"
SOURCE = ROOT / "results/pc_causal_joint_prefix_gradient_20261103_DEV01.json"
STATS = ROOT / "results/pc_pointmaze_affine_nonprogress_lookahead_20261103_DEV01.json"
RESULT_SHA = "eae9fae8ec31448a3eb9722f5e58f70f6ca2f094b1b2b78742f607d2ce1ca295"
STATS_SHA = "a4c1c2a7083195e4951eb26ed559d90af583d6b414eb06125d628c0e9ea17f71"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def verify(report, source, stats, arrays, original):
    require(report["root"] == 8349 and report["model_before"] == report["model_after"], "report identity")
    root = next(r for r in source["rows"] if r["root"] == 8349)
    plans = original["r8349_plans"]
    indices = {"zero": (0, root["control_iterations"][0]),
               "greedy": (1, root["control_iterations"][1]),
               "raw_qualified": tuple(root["decisions"][0]["qualified"])}
    mean = np.asarray(stats["action_mean"], dtype=np.float32)
    std = np.asarray(stats["action_std"], dtype=np.float32)
    require(mean.shape == std.shape == (2,) and np.all(std > 0), "normalizer")
    goal = np.array([1.8, 2.8], dtype=np.float64)
    require(len(report["rows"]) == 9, "row count")
    for i, row in enumerate(report["rows"]):
        prefix, mode = row["prefix"], row["mode"]
        require(prefix in indices and mode in ("old", "zero_init", "carry_init"), "branch identity")
        require((prefix, mode) == (list(indices)[i // 3], ("old", "zero_init", "carry_init")[i % 3]), "row order")
        candidate, iteration = indices[prefix]
        plan = plans[iteration, candidate]
        actions = arrays[f"{prefix}_{mode}_actions"]
        history = arrays[f"{prefix}_{mode}_history"]
        src_actions = original[f"r8349_{prefix}_actions"]
        src_history = original[f"r8349_{prefix}_history"]
        require(np.array_equal(plan[:5], src_actions[2:7]), "source prefix")
        require(np.array_equal(actions[:7], src_actions[:7]), "executed prefix")
        require(np.array_equal(history[:8], src_history[:8]), "prefix RGB")
        require(np.array_equal(actions, arrays["sequences"][i]), "sequence binding")
        if mode == "old":
            require(np.array_equal(actions[7:], plan[5:]), "old tail")
        else:
            base = f"{prefix}_{mode}_feedback3"
            candidates = arrays[base + "_candidates"]
            scores = arrays[base + "_scores"]
            require(candidates.shape == (1, 2, 1, 20) and scores.shape == (2,), "final batch shape")
            require(float(scores[0]) == row["prior_score"] and float(scores[1]) == row["refined_score"], "scores")
            winner = int(scores[1] < scores[0])
            require(bool(winner) == row["refined"], "selection")
            decoded = np.clip(candidates[0, winner, 0].reshape(10, 2) * std + mean, -1, 1).astype(np.float32)
            require(np.array_equal(decoded, actions[7:]), "decoded tail")
            if mode == "zero_init":
                require(np.array_equal(actions, src_actions) and np.array_equal(history, src_history), "source replay")
        require(bool(np.array_equal(actions[7:], plan[5:])) == row["same_as_old"], "old-tail flag")
        for step in (2, 7, 17):
            position = arrays[f"cp{step}_eval_only"][i]
            distance = float(np.linalg.norm(position[:2] - goal))
            require(abs(distance - row[f"cp{step}_distance"]) < 1e-12, "distance")
    return {"branches": 9, "feedback_decisions": 6, "status": "passed"}


def main():
    require(sha(RESULT) == RESULT_SHA and sha(STATS) == STATS_SHA, "receipt hash")
    report, source, stats = (json.loads(p.read_text()) for p in (RESULT, SOURCE, STATS))
    require(sha(SOURCE) == report["source_sha256"], "source hash")
    require(sha(RESULT.with_suffix(".npz")) == report["arrays_sha256"], "result arrays hash")
    require(sha(SOURCE.with_suffix(".npz")) == source["arrays_sha256"], "source arrays hash")
    with np.load(RESULT.with_suffix(".npz"), allow_pickle=False) as arrays, \
            np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as original:
        print(json.dumps(verify(report, source, stats, arrays, original), sort_keys=True))


if __name__ == "__main__":
    main()
