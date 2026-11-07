"""Post-hoc exact-wall candidate crossing and score landscape audit."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_exact_action_identity_retention_dev_20261106_codex as prior
from experiments import pc_exact_temporal_tail_dev_20261107_codex as temporal

OUTPUT = temporal.ROOT / "results/pc_exact_wall_candidate_crossing_audit_20261107_codex.json"


def summarize(task, chains, scores):
    chains = np.asarray(chains, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if chains.ndim != 3 or chains.shape[1:] != (15, 2) or scores.shape != (len(chains),):
        raise ValueError("invalid candidate bank")
    padded = np.pad(chains, ((0, 0), (0, 1), (0, 0))) * prior.exact.wall.HIGH
    trajectory, blocked = prior.exact.wall.rollout(task["start"], padded)
    endpoint = trajectory[:, 15, :2]
    recomputed = np.linalg.norm(endpoint - np.asarray(task["goal"]), axis=1)
    np.testing.assert_allclose(scores, recomputed, rtol=0, atol=1e-12)
    crossed = (trajectory[:, 1:16, 0] > 0).any(axis=1)
    final_right = endpoint[:, 0] > 0
    best = int(np.argmin(scores))
    return {"count": len(chains), "crossed_count": int(crossed.sum()),
            "final_right_count": int(final_right.sum()),
            "any_blocked_count": int(blocked[:, :15].any(axis=1).sum()),
            "best_crossed_distance": float(scores[crossed].min()) if crossed.any() else None,
            "best_uncrossed_distance": float(scores[~crossed].min()) if (~crossed).any() else None,
            "best_distance": float(scores[best]), "best_crossed": bool(crossed[best]),
            "best_endpoint": endpoint[best].tolist(),
            "largest_abs_y": float(np.abs(trajectory[:, :16, 1]).max())}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    trial = json.loads(temporal.OUTPUT.read_text())
    baseline = json.loads(temporal.SOURCE.read_text())
    if trial["dependencies"] != {p: prior.sha(p) for p in trial["dependencies"]}:
        raise ValueError("temporal dependencies changed")
    if baseline["dependencies"] != {p: prior.sha(p) for p in baseline["dependencies"]}:
        raise ValueError("baseline dependencies changed")
    rows = []
    for trial_row, base_row in zip(trial["rows"], baseline["rows"], strict=True):
        if trial_row["root"] != base_row["root"] or trial_row["task"] != base_row["task"]:
            raise ValueError("root mismatch")
        arms = {}
        for name, searches in (("per_step", base_row["pool_longer"]), ("temporal", trial_row["temporal"])):
            chains = np.concatenate([np.asarray(s["actions"]) for s in searches])
            costs = np.concatenate([np.asarray(s["costs"]) for s in searches])
            arms[name] = summarize(trial_row["task"], chains, costs)
        rows.append({"root": trial_row["root"], "arms": arms})
    summary = {}
    for arm in ("per_step", "temporal"):
        summaries = [r["arms"][arm] for r in rows]
        summary[arm] = {"candidates": sum(s["count"] for s in summaries),
                        "crossed": sum(s["crossed_count"] for s in summaries),
                        "roots_with_crossing": sum(s["crossed_count"] > 0 for s in summaries),
                        "roots_best_crossed": sum(s["best_crossed"] for s in summaries),
                        "blocked_candidates": sum(s["any_blocked_count"] for s in summaries),
                        "largest_abs_y": max(s["largest_abs_y"] for s in summaries)}
    record = {"rows": rows, "summary": summary, "baseline_sha256": prior.sha(temporal.SOURCE),
              "temporal_sha256": prior.sha(temporal.OUTPUT), "audit_sha256": prior.sha(__file__),
              "scope": "post-hoc exposed exact-wall evaluation; not deployable planner evidence"}
    with OUTPUT.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
