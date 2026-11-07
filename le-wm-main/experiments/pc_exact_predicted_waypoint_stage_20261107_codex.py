"""Exact-state development: condition stage transition on predicted path."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_prefix_successor_composition_20261107_codex as base

PROTOCOL = base.ROOT.parent / "docs/PC_EXACT_PREDICTED_WAYPOINT_STAGE_PROTOCOL_20261107_CODEX.md"
OUT = base.ROOT / "results/pc_exact_predicted_waypoint_stage_20261107_codex.json"


def reached_waypoint(trace, nominal_step, waypoint):
    trace, waypoint = np.asarray(trace, float), np.asarray(waypoint, float)
    if (trace.shape != (6, 3) or waypoint.shape != (2,) or not np.isfinite(trace).all()
            or not np.isfinite(waypoint).all() or not isinstance(nominal_step, (int, np.integer))
            or isinstance(nominal_step, (bool, np.bool_)) or nominal_step < 0):
        raise ValueError("five-step finite prediction and nonnegative integer stage required")
    return bool(nominal_step <= 5 and np.linalg.norm(trace[nominal_step, :2] - waypoint) <= 1e-9)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prior = json.loads(base.OUTPUT.read_text())
    source = json.loads(base.prior.OUTPUT.read_text())
    if prior["dependencies"] != {p: base.sha(p) for p in prior["dependencies"]}:
        raise ValueError("prior dependencies changed")
    if len(prior["rows"]) != 24 or len(source["rows"]) != 24:
        raise ValueError("panel mismatch")
    paths = [Path(__file__), PROTOCOL, base.OUTPUT, base.prior.OUTPUT, Path(base.__file__),
             Path(base.wall.__file__), Path(base.wall.physics.__file__)]
    dependencies = {str(p): base.sha(p) for p in paths}
    print("Exact-state development: predicted-trajectory stage vs nominal stage, matched actions/budget.", flush=True)
    rows = []
    for old, original in zip(prior["rows"], source["rows"], strict=True):
        if old["root"] != original["root"] or old["task"] != original["task"]:
            raise ValueError("task mismatch")
        task = old["task"]
        start, goal = np.asarray(task["start"], float), np.asarray(task["goal"], float)
        prefix = np.asarray(old["prefix_actions"], float)
        if prefix.shape != (192, 5, 2):
            raise ValueError("prefix mismatch")
        trajectory, blocked = base.rollout_n(start, prefix)
        np.testing.assert_allclose(trajectory[:, -1], old["prefix_successors"], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(blocked, old["prefix_blocked"])
        originals = [base.prior.make_chain(start, goal, offset) for offset in original["offsets"][:192]]
        flags, chains = [], []
        for i, (_, waypoint, _) in enumerate(originals):
            n = old["waypoint_stage_steps"][i]
            _, counts = base.route(start, [waypoint])
            if n != counts[0]:
                raise ValueError("stage duration drift")
            flag = reached_waypoint(trajectory[i], n, waypoint)
            tail, _ = base.suffix_from_successor(trajectory[i, -1], waypoint, goal, flag)
            flags.append(flag)
            chains.append(np.concatenate((prefix[i], tail)))
        chains = np.stack(chains)
        trace, _ = base.rollout_n(start, chains)
        values = np.linalg.norm(trace[:, -1, :2] - goal, axis=1)
        np.testing.assert_allclose(trace[:, -1, :2], base.prior.baseline.prior.exact.endpoints(task, chains),
                                   atol=1e-12, rtol=0)
        chosen = int(np.argmin(values))
        old_flags = np.asarray(old["waypoint_stage_steps"]) <= 5
        rows.append({"root": old["root"], "task": task, "flags": flags,
                     "nominal_true_but_not_reached": int(np.sum(old_flags & ~np.asarray(flags))),
                     "old_cost": old["costs"][old["chosen"]], "baseline_cost": old["base_cost"],
                     "new_costs": values.tolist(), "new_actions": chains.tolist(), "chosen": chosen,
                     "model_steps": 3840})
        print(f"root={old['root']} old={old['costs'][old['chosen']]:.5f} new={values[chosen]:.5f}", flush=True)
    if dependencies != {p: base.sha(p) for p in dependencies}:
        raise ValueError("dependency changed")
    with OUT.open("x") as stream:
        json.dump({"dependencies": dependencies, "rows": rows,
                   "scope": "exposed exact-state developer correction, not RGB or descriptor evidence"},
                  stream, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
