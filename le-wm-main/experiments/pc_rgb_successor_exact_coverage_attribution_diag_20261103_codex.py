"""Analyze coverage and retention in sealed exact successor prediction traces."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_rgb_successor_exact_same_policy_dev_20261103_codex as exp
from experiments import pc_rgb_successor_exact_collision_attribution_diag_20261103_codex as collision

PROTOCOL = exp.ROOT.parent / "docs/PC_RGB_SUCCESSOR_EXACT_COVERAGE_ATTRIBUTION_DIAG_PROTOCOL_20261103_CODEX.md"
OUTPUT = exp.PREFIX.with_name(exp.PREFIX.name + "_coverage_attribution_diagnostic.json")


def evaluate(task, actions):
    trace, blocked = collision.rollout(task, actions)
    horizon = np.asarray(actions).shape[1]
    return trace[:, horizon, :2], np.any(blocked[:, :horizon], axis=1)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plans = json.loads(exp.PLANS.read_text())
    audit = json.loads(exp.AUDIT.read_text())
    if plans["dependencies"] != exp.dependencies() or audit["plans_sha256"] != exp.sha(exp.PLANS):
        raise ValueError("source or receipt drift")
    rows = []
    for row in plans["rows"]:
        task, result = row["task"], row["result"]
        first = np.asarray([x["prefix"] for x in result["first_bank"]])
        second = np.asarray([x["prefix"] for x in result["bank"]])
        first_pos, first_blocked = evaluate(task, first)
        second_pos, second_blocked = evaluate(task, second)
        n_full = n_far = n_blocked = n_success = 0
        best, best_actions = float("inf"), None
        for batch in result["prediction_trace"]:
            requested = np.asarray(batch["requested_actions"])
            actions = np.asarray(batch["model_actions"], dtype=np.float32)
            np.testing.assert_array_equal(requested.astype(np.float32), actions)
            positions, blocked = evaluate(task, actions)
            values = np.linalg.norm(positions - task["goal"], axis=1)
            np.testing.assert_allclose(positions, batch["positions"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(values, batch["scores"], rtol=0, atol=1e-12)
            if actions.shape[1] != 15:
                continue
            n_full += len(actions)
            n_far += int(np.count_nonzero(positions[:, 0] > 0))
            n_blocked += int(np.count_nonzero(blocked))
            n_success += int(np.count_nonzero(values < .05))
            i = int(np.argmin(values))
            if values[i] < best:
                best, best_actions = float(values[i]), requested[i].copy()
        selected = np.asarray(result["selected"]["chain"])
        selected_score = float(result["selected"]["terminal_score"])
        rows.append({"root": row["root"], "first_count": len(first),
                     "first_far": int(np.count_nonzero(first_pos[:, 0] > 0)),
                     "first_blocked": int(np.count_nonzero(first_blocked)),
                     "second_count": len(second), "second_far": int(np.count_nonzero(second_pos[:, 0] > 0)),
                     "second_blocked": int(np.count_nonzero(second_blocked)),
                     "full_count": n_full, "full_far": n_far, "full_blocked": n_blocked,
                     "full_success": n_success, "best_score": best,
                     "best_requested_chain": best_actions.tolist(), "selected_score": selected_score,
                     "selected_equals_best_requested": bool(np.array_equal(selected, best_actions)),
                     "selection_minus_best": selected_score - best})
    with OUTPUT.open("x") as stream:
        json.dump({"plans_sha256": exp.sha(exp.PLANS), "audit_sha256": exp.sha(exp.AUDIT),
                   "protocol_sha256": exp.sha(PROTOCOL), "runner_sha256": exp.sha(Path(__file__)),
                   "rows": rows}, stream, indent=2, allow_nan=False)
    print(f"Audited generation, samples, and final selection for {len(rows)} roots.", flush=True)


if __name__ == "__main__":
    main()
