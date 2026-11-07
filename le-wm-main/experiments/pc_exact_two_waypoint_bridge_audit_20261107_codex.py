"""Replay every new action chain and its exact-wall trajectory."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_two_waypoint_bridge_20261107_codex as run

OUTPUT = run.ROOT / "results/pc_exact_two_waypoint_bridge_audit_20261107_codex.json"


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads(run.OUTPUT.read_text())
    old = json.loads(run.prior.OUTPUT.read_text())
    if result["dependencies"] != {p: run.sha(p) for p in result["dependencies"]}:
        raise ValueError("dependency mismatch")
    summaries = []
    for row, baseline in zip(result["rows"], old["rows"], strict=True):
        if row["root"] != baseline["root"] or row["task"] != baseline["task"]:
            raise ValueError("paired task mismatch")
        task = row["task"]
        chains = np.asarray(row["actions"])
        if chains.shape != (384, 15, 2) or not np.isfinite(chains).all() or np.max(np.abs(chains)) > 1:
            raise ValueError("invalid actions")
        for i, offset in enumerate(baseline["offsets"]):
            chain, points, required = run.make_chain(task["start"], task["goal"], offset)
            np.testing.assert_array_equal(chains[i], chain)
            np.testing.assert_allclose(row["points"][i], points, atol=1e-12, rtol=0)
            if required != row["required_steps"][i]:
                raise ValueError("duration mismatch")
        padded = np.pad(chains, ((0, 0), (0, 1), (0, 0))) * run.wall.HIGH
        trajectory, blocked = run.wall.rollout(task["start"], padded)
        cost = np.linalg.norm(trajectory[:, 15, :2] - task["goal"], axis=1)
        np.testing.assert_allclose(cost, row["costs"], atol=1e-12, rtol=0)
        selected = int(np.argmin(cost))
        if selected != row["best"] or row["evaluations"] != 384:
            raise ValueError("selection or budget mismatch")
        old_cost = baseline["costs"][baseline["best"]]
        summaries.append({"root": row["root"], "score": float(cost[selected]), "old_score": old_cost,
                          "success": bool(cost[selected] < .05), "old_success": old_cost < .05,
                          "blocked_selected": int(blocked[selected, :15].sum()),
                          "selected_required_steps": row["required_steps"][selected],
                          "truncated": int(np.sum(np.asarray(row["required_steps"]) > 15)),
                          "crossed_candidates": int((trajectory[:, 1:16, 0] > 0).any(axis=1).sum()),
                          "seconds": row["seconds"], "old_seconds": baseline["seconds"]})
    wins = sum(r["success"] and not r["old_success"] for r in summaries)
    losses = sum(r["old_success"] and not r["success"] for r in summaries)
    summary = {"success": sum(r["success"] for r in summaries),
               "old_success": sum(r["old_success"] for r in summaries),
               "new_only": wins, "old_only": losses,
               "mean_distance": float(np.mean([r["score"] for r in summaries])),
               "old_mean_distance": float(np.mean([r["old_score"] for r in summaries])),
               "truncated_candidates": sum(r["truncated"] for r in summaries),
               "crossed_candidates": sum(r["crossed_candidates"] for r in summaries),
               "mean_seconds": float(np.mean([r["seconds"] for r in summaries])),
               "old_mean_seconds": float(np.mean([r["old_seconds"] for r in summaries])),
               "development_exit": wins - losses >= 4 and losses <= 2}
    with OUTPUT.open("x") as stream:
        json.dump({"source_sha256": run.sha(run.OUTPUT), "audit_sha256": run.sha(Path(__file__)),
                   "summary": summary, "roots": summaries}, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    audit()
