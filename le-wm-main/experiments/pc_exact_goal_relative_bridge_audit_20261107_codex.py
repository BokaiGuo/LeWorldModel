"""Replay and audit complete waypoint proposals and paired fixed-budget baseline."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_goal_relative_bridge_20261107_codex as run

OUTPUT = run.ROOT / "results/pc_exact_goal_relative_bridge_audit_20261107_codex.json"
OLD_AUDIT = run.ROOT / "results/pc_exact_block_chain_audit_20261107_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads(run.OUTPUT.read_text())
    old = json.loads(run.baseline.OUTPUT.read_text())
    old_audit = json.loads(OLD_AUDIT.read_text())
    if (result["dependencies"] != {p: run.baseline.prior.sha(p) for p in result["dependencies"]}
            or old_audit["record_sha256"] != run.baseline.prior.sha(run.baseline.OUTPUT)):
        raise ValueError("source or audited baseline changed")
    summaries = []
    for row, prior in zip(result["rows"], old["rows"], strict=True):
        root, task = row["root"], row["task"]
        if root != prior["root"] or task != prior["task"] or row["tokens"] != 1152:
            raise ValueError("paired task or budget mismatch")
        offsets = np.random.default_rng(2713000 + root).uniform(-2., 2., 384)
        np.testing.assert_array_equal(offsets, row["offsets"])
        chains = np.asarray(row["actions"])
        if chains.shape != (384, 15, 2) or np.any(np.abs(chains) > 1 + 1e-12):
            raise ValueError("invalid action chain")
        for i, offset in enumerate(offsets):
            chain, waypoint, required = run.make_chain(task["start"], task["goal"], offset)
            np.testing.assert_array_equal(chain, chains[i])
            np.testing.assert_allclose(waypoint, row["waypoints"][i], atol=1e-12, rtol=0)
            if required != row["required_steps"][i]:
                raise ValueError("program duration mismatch")
        costs = np.linalg.norm(run.baseline.prior.exact.endpoints(task, chains) - task["goal"], axis=1)
        np.testing.assert_allclose(costs, row["costs"], atol=1e-12, rtol=0)
        best = int(np.argmin(costs))
        if best != row["best"]:
            raise ValueError("selection mismatch")
        padded = np.pad(chains, ((0, 0), (0, 1), (0, 0))) * run.baseline.prior.exact.wall.HIGH
        trajectory, _ = run.baseline.prior.exact.wall.rollout(task["start"], padded)
        crossed = (trajectory[:, 1:16, 0] > 0).any(axis=1)
        base = prior["arms"]["step"]
        summaries.append({"root": root, "score": float(costs[best]), "success": bool(costs[best] < .05),
                          "baseline_score": float(base["costs"][base["best"]]),
                          "crossing_candidates": int(crossed.sum()), "selected_crossed": bool(crossed[best]),
                          "truncated": int(np.sum(np.asarray(row["required_steps"]) > 15)),
                          "seconds": row["seconds"], "baseline_seconds": base["seconds"]})
    success = sum(r["success"] for r in summaries)
    baseline_success = sum(r["baseline_score"] < .05 for r in summaries)
    summary = {"success": success, "baseline_success": baseline_success,
               "goal_only_success": sum(r["success"] and r["baseline_score"] >= .05 for r in summaries),
               "baseline_only_success": sum(not r["success"] and r["baseline_score"] < .05 for r in summaries),
               "mean_distance": float(np.mean([r["score"] for r in summaries])),
               "baseline_mean_distance": float(np.mean([r["baseline_score"] for r in summaries])),
               "crossing_candidates": sum(r["crossing_candidates"] for r in summaries),
               "crossing_roots": sum(r["crossing_candidates"] > 0 for r in summaries),
               "selected_crossing_roots": sum(r["selected_crossed"] for r in summaries),
               "truncated_candidates": sum(r["truncated"] for r in summaries),
               "mean_seconds": float(np.mean([r["seconds"] for r in summaries])),
               "baseline_mean_seconds": float(np.mean([r["baseline_seconds"] for r in summaries])),
               "development_exit": success >= 4 and success - baseline_success >= 3}
    with OUTPUT.open("x") as stream:
        json.dump({"source_sha256": run.baseline.prior.sha(run.OUTPUT),
                   "baseline_audit_sha256": run.baseline.prior.sha(OLD_AUDIT),
                   "audit_sha256": run.baseline.prior.sha(Path(__file__)),
                   "summary": summary, "roots": summaries}, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
