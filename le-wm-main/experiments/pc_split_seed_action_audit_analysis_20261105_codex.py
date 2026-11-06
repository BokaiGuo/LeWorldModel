"""Fail-closed paired reset and actual action-domain audit."""
import json
from pathlib import Path

import numpy as np


def load(trace_path):
    groups = {}
    for line in trace_path.read_text().splitlines():
        row = json.loads(line)
        seed = row["reset_seed"]
        groups.setdefault(seed, []).append(row)
    for seed, rows in groups.items():
        if [row["step"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"missing or duplicate steps for reset seed {seed}")
    return groups


def summarize(groups):
    count = 0
    tasks = 0
    largest = 0.0
    for rows in groups.values():
        actions = np.asarray([row["action"] for row in rows], dtype=np.float64)
        if not np.isfinite(actions).all():
            raise ValueError("nonfinite action")
        violating = np.any(np.abs(actions) > 1.0, axis=1)
        count += int(violating.sum())
        tasks += int(violating.any())
        largest = max(largest, float(np.abs(actions).max()))
    return {"n_tasks": len(groups), "executed_steps": sum(map(len, groups.values())),
            "out_of_bounds_actions": count, "tasks_with_out_of_bounds_actions": tasks,
            "maximum_absolute_action": largest}


def run(root):
    seeds = (42, 1042, 2042)
    groups = {}
    metrics = {}
    for seed in seeds:
        name = f"pc_split_seed_action_audit_task42_solver{seed}_20261105_codex"
        groups[seed] = load(root / f"{name}.jsonl")
        metrics[seed] = json.loads((root / name / "eval_metrics.json").read_text())
    reference = groups[42]
    for seed in seeds:
        if groups[seed].keys() != reference.keys() or len(reference) != 50:
            raise ValueError("task/reset identities differ or task count is not 50")
        for key in reference:
            if not np.array_equal(reference[key][0]["before"], groups[seed][key][0]["before"]):
                raise ValueError(f"initial physical snapshot differs: solver {seed} task {key}")
    result = {"initial_snapshot_parity": True, "task_seed": 42, "reset_seed": 771001,
              "arms": {seed: {**summarize(groups[seed]), "metrics": metrics[seed]}
                       for seed in seeds}}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1] / "results")
