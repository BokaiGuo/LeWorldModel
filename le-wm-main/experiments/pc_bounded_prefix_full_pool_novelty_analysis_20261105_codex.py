"""Summarize full-pool novelty without treating correlated contexts as tasks."""
import json
from pathlib import Path

import numpy as np

from experiments.pc_split_seed_action_audit_analysis_20261105_codex import load, summarize


def main():
    root = Path(__file__).resolve().parents[1] / "results"
    name = "pc_bounded_prefix_full_pool_novelty_seed42_20261105_codex"
    runs = [json.loads(line) for line in (root / f"{name}.audit.jsonl").read_text().splitlines()]
    trace = load(root / f"{name}.trace.jsonl")
    baseline = load(root / "pc_bounded_cem_task42_solver42_20261105_codex.trace.jsonl")
    if len(trace) != 50 or trace.keys() != baseline.keys():
        raise ValueError("task identities mismatch")
    if any(not np.array_equal(trace[key][0]["before"], baseline[key][0]["before"]) for key in trace):
        raise ValueError("initial physical snapshots do not match")
    action_stats = summarize(trace)
    if action_stats["out_of_bounds_actions"]:
        raise ValueError("bounded solver produced illegal executed action")
    contexts = novel_contexts = novel_entries = novel_elites = improved = 0
    extra_scores = 0
    per_task = set()
    for run in runs:
        for row in run["iterations"]:
            full = row["full_pool_audit"]
            if full is None:
                continue
            for offset, record in enumerate(full["records"]):
                contexts += 1
                extra_scores += full["additional_full_chain_scores_per_environment"]
                unique, elite = record["unique_mask"], record["proposal_elite_mask"]
                n = sum(unique)
                novel_entries += n
                novel_elites += sum(u and e for u, e in zip(unique, elite))
                if n:
                    novel_contexts += 1
                    if run["solve"] == 0:
                        per_task.add(row["batch_start"] + offset)
                    if record["best_unique"]["cost_minus_pool_best"] < 0:
                        improved += 1
    metrics = json.loads((root / name / "eval_metrics.json").read_text())
    print(json.dumps({"runs": len(runs), "task_count": len(trace), "contexts": contexts,
                      "novel_contexts": novel_contexts, "novel_entries": novel_entries,
                      "novel_elite_entries": novel_elites, "novel_better_than_full_pool_contexts": improved,
                      "first_solve_tasks_with_novel": len(per_task),
                      "additional_full_chain_scores": extra_scores,
                      "success_count": sum(metrics["episode_successes"]),
                      "physical_initial_parity": True, "action_stats": action_stats},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
