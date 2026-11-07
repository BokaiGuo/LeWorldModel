"""Reconstruct actions and audit all model-step traces for the detour experiment."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_blocked_detour_20261107_codex as run

OUTPUT = run.ROOT / "results/pc_exact_blocked_detour_audit_20261107_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    record = json.loads(run.OUTPUT.read_text())
    source = json.loads(run.SOURCE.read_text())
    if record["dependencies"] != {p: run.base.sha(p) for p in record["dependencies"]}:
        raise ValueError("dependency drift")
    summary = {"roots": 0, "chains": 0, "model_steps": 0, "triggers": 0,
               "selected_with_trigger": 0, "old_success": 0, "new_success": 0,
               "wins": 0, "losses": 0, "max_endpoint_error": 0.}
    for saved, old in zip(record["rows"], source["rows"], strict=True):
        if saved["root"] != old["root"] or saved["task"] != old["task"]:
            raise ValueError("task mismatch")
        actions = np.asarray(saved["actions"])
        rebuilt, trace, trigger, count = run.generate(old["task"]["start"], old["task"]["goal"], old["offsets"])
        np.testing.assert_array_equal(actions, rebuilt)
        np.testing.assert_array_equal(trigger, saved["trigger_steps"])
        np.testing.assert_array_equal(count, saved["trigger_counts"])
        if actions.shape != (384, 15, 2) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
            raise ValueError("invalid candidate")
        displacement = np.linalg.norm(np.diff(trace[:, :, :2], axis=1), axis=2)
        inferred = (actions[:, :, 0] > 0) & (displacement < 1e-9)
        np.testing.assert_array_equal(inferred, trigger)
        endpoint = run.base.prior.baseline.prior.exact.endpoints(old["task"], actions)
        summary["max_endpoint_error"] = max(summary["max_endpoint_error"],
                                             float(np.max(np.abs(endpoint - trace[:, -1, :2]))))
        np.testing.assert_allclose(endpoint, trace[:, -1, :2], atol=1e-12, rtol=0)
        costs = np.linalg.norm(endpoint - old["task"]["goal"], axis=1)
        np.testing.assert_allclose(costs, saved["costs"], atol=1e-12, rtol=0)
        chosen = int(np.argmin(costs))
        if chosen != saved["chosen"] or saved["model_steps"] != 384 * 15:
            raise ValueError("selection or budget mismatch")
        old_value = float(old["costs"][old["best"]])
        if abs(saved["old_best"] - old_value) > 1e-12 or abs(saved["new_best"] - costs[chosen]) > 1e-12:
            raise ValueError("best cost mismatch")
        old_success, new_success = old_value < .05, costs[chosen] < .05
        summary["roots"] += 1
        summary["chains"] += len(actions)
        summary["model_steps"] += len(actions) * actions.shape[1]
        summary["triggers"] += int(count.sum())
        summary["selected_with_trigger"] += int(count[chosen] > 0)
        summary["old_success"] += int(old_success)
        summary["new_success"] += int(new_success)
        summary["wins"] += int(new_success and not old_success)
        summary["losses"] += int(old_success and not new_success)
    if summary["roots"] != 24:
        raise ValueError("incomplete panel")
    with OUTPUT.open("x") as stream:
        json.dump({"summary": summary, "result_sha256": run.base.sha(run.OUTPUT),
                   "auditor_sha256": run.base.sha(Path(__file__)),
                   "scope": "same generator and physics backend, separate endpoint implementation"},
                  stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
