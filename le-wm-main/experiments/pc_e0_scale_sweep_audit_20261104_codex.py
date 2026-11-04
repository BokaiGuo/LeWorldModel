"""Recompute the saved scale sweep from source actions and independent geometry."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_e0_response_waypoint_direction_dev_20261104_codex as source
from experiments.pc_e0_wall_independent_branch_replay_dev_20261104_codex import independent
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

INPUT = source.OUT / "scale_sweep_posthoc.json"
OUTPUT = source.OUT / "scale_sweep_posthoc_audit.json"
SCALES = (0., .05, .1, .25, .5, 1.)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sweep = json.loads(INPUT.read_text())
    seal.verify(sweep["dependencies"])
    original = json.loads((source.OUT / "result.json").read_text())
    original_rows = {(r["root"], r["arm"]): r for r in original["rows"]
                     if r["arm"].endswith("_waypoint")}
    assert len(sweep["rows"]) == 48
    seen = set()
    by_scale = [[] for _ in SCALES]
    for row in sweep["rows"]:
        key = (row["root"], row["arm"])
        original_row = original_rows[key]
        identity = (*key, row["branch"])
        assert identity not in seen
        seen.add(identity)
        with np.load(source.OUT / original_row["trace_file"], allow_pickle=False) as trace:
            nominal = trace["actions_2"][0].astype(np.float32)
            endpoint0 = trace["positions_2"][0]
        prefix = np.asarray(original_row["result"]["waypoints"][row["branch"]]["prefix"], np.float32)
        np.testing.assert_array_equal(nominal, np.asarray(row["nominal"], np.float32))
        np.testing.assert_array_equal(prefix, np.asarray(row["prefix"], np.float32))
        matrix = np.asarray(original_row["result"]["response"]["matrix"])
        start, goal = original_row["task"]["start"], np.asarray(original_row["task"]["goal"])
        delta = prefix.astype(float) - nominal.astype(float)
        actions = np.asarray([nominal.astype(float) + a * delta for a in SCALES], np.float32)
        path, blocked = independent(start, actions)
        for i, saved in enumerate(row["scales"]):
            actual = path[i, -1, :2]
            predicted = endpoint0 + matrix @ (actions[i].astype(float) - nominal).ravel()
            np.testing.assert_allclose(saved["endpoint"], actual, atol=1e-12, rtol=0)
            np.testing.assert_allclose(saved["predicted_endpoint"], predicted, atol=1e-12, rtol=0)
            abs_error = float(np.linalg.norm(actual - predicted))
            displacement = float(np.linalg.norm(actual - endpoint0))
            np.testing.assert_allclose(saved["absolute_error"], abs_error, atol=1e-12, rtol=0)
            np.testing.assert_allclose(saved["displacement"], displacement, atol=1e-12, rtol=0)
            assert saved["branch_changed"] == bool(np.any(blocked[i] != blocked[0]))
            if displacement > 1e-10:
                np.testing.assert_allclose(saved["relative_error"], abs_error / displacement, atol=1e-12)
            else:
                assert saved["relative_error"] is None
            physical_change = np.linalg.norm(actual - goal) - np.linalg.norm(endpoint0 - goal)
            predicted_change = np.linalg.norm(predicted - goal) - np.linalg.norm(endpoint0 - goal)
            np.testing.assert_allclose(saved["goal_change"], physical_change, atol=1e-12)
            np.testing.assert_allclose(saved["predicted_goal_change"], predicted_change, atol=1e-12)
            sign = (bool(np.sign(physical_change) == np.sign(predicted_change))
                    if min(abs(physical_change), abs(predicted_change)) > 1e-10 else None)
            assert saved["sign_agreement"] == sign
            by_scale[i].append((row["root"], saved))
    root_summary = []
    for i, alpha in enumerate(SCALES):
        items = by_scale[i]
        roots = sorted({root for root, _ in items})
        root_medians = []
        for root in roots:
            values = [s["relative_error"] for r, s in items
                      if r == root and s["relative_error"] is not None]
            if values:
                root_medians.append(float(np.median(values)))
        saved = sweep["summary"][i]
        assert saved["alpha"] == alpha
        np.testing.assert_allclose(saved["median_absolute_error"],
                                   np.median([s["absolute_error"] for _, s in items]), atol=1e-12)
        assert saved["changed_branches"] == sum(s["branch_changed"] for _, s in items)
        assert saved["goal_sign_agree"] == sum(s["sign_agreement"] is True for _, s in items)
        root_summary.append({"alpha": alpha, "roots": len(roots),
                             "root_median_relative_error_range": ([min(root_medians), max(root_medians)]
                                                                   if root_medians else None),
                             "roots_with_branch_change": len({r for r, s in items if s["branch_changed"]})})
    dependencies = seal.capture([Path(__file__), INPUT])
    source.prior.previous.write(OUTPUT, {"status": "passed", "input_sha256": seal.digest(INPUT),
        "dependencies": dependencies, "directions": len(seen), "root_summary": root_summary,
        "scope": "same-backend source-action and arithmetic recomputation; not independent replication"})
    print(json.dumps(root_summary, indent=2))


if __name__ == "__main__":
    main()
