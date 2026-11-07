"""Replay every recorded candidate and calculate paired development summary."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_prefix_successor_composition_20261107_codex as run

OUTPUT = run.ROOT / "results/pc_exact_prefix_successor_composition_audit_20261107_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads(run.OUTPUT.read_text())
    if result["dependencies"] != {p: run.sha(p) for p in result["dependencies"]}:
        raise ValueError("dependency drift")
    old = json.loads(run.prior.OUTPUT.read_text())
    summaries = []
    for row, source in zip(result["rows"], old["rows"], strict=True):
        if row["root"] != source["root"] or row["task"] != source["task"]:
            raise ValueError("task mismatch")
        task, start = row["task"], np.asarray(row["task"]["start"])
        goal = np.asarray(task["goal"])
        original = [run.prior.make_chain(start, goal, offset) for offset in source["offsets"][:256]]
        base = np.stack([p[0] for p in original])
        prefixes = np.asarray(row["prefix_actions"])
        chains = np.asarray(row["actions"])
        if (prefixes.shape != (192, 5, 2) or chains.shape != (192, 15, 2)
                or np.any(np.abs(chains) > 1 + 1e-12)):
            raise ValueError("candidate or action bound mismatch")
        np.testing.assert_array_equal(prefixes, base[:192, :5])
        padded = np.pad(prefixes, ((0, 0), (0, 11), (0, 0))) * run.wall.HIGH
        prior_trajectory, prefix_events = run.wall.rollout(start, padded)
        successor = prior_trajectory[:, 5]
        np.testing.assert_allclose(successor, row["prefix_successors"], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(prefix_events[:, :5], row["prefix_blocked"])
        for i, (_, waypoint, _) in enumerate(original[:192]):
            _, counts = run.route(start, [waypoint])
            if counts[0] != row["waypoint_stage_steps"][i]:
                raise ValueError("waypoint stage mismatch")
            tail, _ = run.suffix_from_successor(successor[i], waypoint, goal, counts[0] <= 5)
            np.testing.assert_allclose(chains[i], np.concatenate((prefixes[i], tail)), rtol=0, atol=1e-12)
        costs = np.linalg.norm(run.prior.baseline.prior.exact.endpoints(task, chains) - goal, axis=1)
        base_costs = np.linalg.norm(run.prior.baseline.prior.exact.endpoints(task, base) - goal, axis=1)
        np.testing.assert_allclose(costs, row["costs"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(base_costs, source["costs"][:256], atol=1e-12, rtol=0)
        if (row["chosen"] != int(np.argmin(costs)) or row["base_index"] != int(np.argmin(base_costs))
                or row["baseline_model_steps"] != 3840 or row["composed_model_steps"] != 3840):
            raise ValueError("selection or budget mismatch")
        np.testing.assert_allclose(row["base_action"], base[row["base_index"]], atol=1e-12, rtol=0)
        np.testing.assert_allclose(row["base_cost"], base_costs.min(), atol=1e-12, rtol=0)
        summaries.append({"root": row["root"], "baseline": float(base_costs.min()),
                          "composed": float(costs.min()),
                          "prefix_blocked": int(prefix_events[:, :5].any(axis=1).sum())})
    base = np.asarray([x["baseline"] for x in summaries])
    composed = np.asarray([x["composed"] for x in summaries])
    b_ok, c_ok = base < .05, composed < .05
    only_c, only_b = int(np.sum(c_ok & ~b_ok)), int(np.sum(b_ok & ~c_ok))
    summary = {"roots": len(summaries), "baseline_success": int(b_ok.sum()),
               "composed_success": int(c_ok.sum()), "composed_only_success": only_c,
               "baseline_only_success": only_b, "baseline_mean_distance": float(base.mean()),
               "composed_mean_distance": float(composed.mean()),
               "composed_strictly_better": int(np.sum(composed < base - 1e-12)),
               "composed_strictly_worse": int(np.sum(composed > base + 1e-12)),
               "prefixes_with_blocked_steps": sum(x["prefix_blocked"] for x in summaries),
               "development_exit": bool(only_c >= 3 and only_b <= 1)}
    with OUTPUT.open("x") as stream:
        json.dump({"summary": summary, "rows": summaries, "source_sha256": run.sha(run.OUTPUT),
                   "audit_sha256": run.sha(Path(__file__)),
                   "scope": "replay same exact backend, not independent physics"}, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
