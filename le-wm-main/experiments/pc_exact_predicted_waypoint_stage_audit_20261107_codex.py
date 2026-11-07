"""Audit predicted waypoint stage and replay all selected alternatives."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_predicted_waypoint_stage_20261107_codex as exp

OUTPUT = exp.base.ROOT / "results/pc_exact_predicted_waypoint_stage_audit_20261107_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    record = json.loads(exp.OUT.read_text())
    if record["dependencies"] != {p: exp.base.sha(p) for p in record["dependencies"]}:
        raise ValueError("dependency drift")
    old = json.loads(exp.base.OUTPUT.read_text())
    source = json.loads(exp.base.prior.OUTPUT.read_text())
    rows = []
    for new, prev, initial in zip(record["rows"], old["rows"], source["rows"], strict=True):
        if new["root"] != prev["root"] or new["task"] != prev["task"] or new["root"] != initial["root"]:
            raise ValueError("task drift")
        task = new["task"]
        start, goal = np.asarray(task["start"]), np.asarray(task["goal"])
        prefix = np.asarray(prev["prefix_actions"])
        padded = np.pad(prefix, ((0, 0), (0, 11), (0, 0))) * exp.base.wall.HIGH
        trajectory, _ = exp.base.wall.rollout(start, padded)
        all_actions = np.asarray(new["new_actions"])
        flags = np.asarray(new["flags"])
        if all_actions.shape != (192, 15, 2) or np.any(np.abs(all_actions) > 1 + 1e-12):
            raise ValueError("invalid action")
        np.testing.assert_array_equal(all_actions[:, :5], prefix)
        nominal = np.asarray(prev["waypoint_stage_steps"])
        for i, offset in enumerate(initial["offsets"][:192]):
            _, waypoint, _ = exp.base.prior.make_chain(start, goal, offset)
            reached = bool(nominal[i] <= 5 and
                           np.linalg.norm(trajectory[i, nominal[i], :2] - waypoint) <= 1e-9)
            if flags[i] != reached:
                raise ValueError("stage classification differs")
            tail, _ = exp.base.suffix_from_successor(trajectory[i, 5], waypoint, goal, reached)
            np.testing.assert_allclose(all_actions[i, 5:], tail, atol=1e-12, rtol=0)
        costs = np.linalg.norm(exp.base.prior.baseline.prior.exact.endpoints(task, all_actions) - goal, axis=1)
        np.testing.assert_allclose(costs, new["new_costs"], atol=1e-12, rtol=0)
        if (new["chosen"] != int(np.argmin(costs)) or new["model_steps"] != 3840
                or new["nominal_true_but_not_reached"] != int(np.sum((nominal <= 5) & ~flags))):
            raise ValueError("selection, budget, or stage count mismatch")
        if new["old_cost"] != prev["costs"][prev["chosen"]] or new["baseline_cost"] != prev["base_cost"]:
            raise ValueError("old result mismatch")
        rows.append({"root": new["root"], "old": new["old_cost"], "new": float(costs.min()),
                     "baseline": new["baseline_cost"], "misclassified": new["nominal_true_but_not_reached"]})
    old_values = np.array([r["old"] for r in rows])
    new_values = np.array([r["new"] for r in rows])
    baseline = np.array([r["baseline"] for r in rows])
    old_ok, new_ok = old_values < .05, new_values < .05
    wins, losses = int(np.sum(new_ok & ~old_ok)), int(np.sum(old_ok & ~new_ok))
    summary = {"roots": len(rows), "old_success": int(old_ok.sum()), "new_success": int(new_ok.sum()),
               "baseline_success": int(np.sum(baseline < .05)), "new_only_success": wins,
               "old_only_success": losses, "new_mean_distance": float(new_values.mean()),
               "old_mean_distance": float(old_values.mean()),
               "distance_improved": int(np.sum(new_values < old_values - 1e-12)),
               "distance_worsened": int(np.sum(new_values > old_values + 1e-12)),
               "nominal_reached_but_predicted_not": sum(r["misclassified"] for r in rows),
               "development_exit": bool(wins >= 3 and losses <= 1 and new_ok.sum() >= np.sum(baseline < .05))}
    with OUTPUT.open("x") as f:
        json.dump({"summary": summary, "rows": rows, "source_sha256": exp.base.sha(exp.OUT),
                   "audit_sha256": exp.base.sha(Path(__file__)),
                   "scope": "replay using original dynamics; not independent physical engine"}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
