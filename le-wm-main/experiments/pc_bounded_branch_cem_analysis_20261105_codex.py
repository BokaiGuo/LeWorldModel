"""Compare two-branch development control with matched bounded single CEM."""
import json
from pathlib import Path

import numpy as np

from experiments.pc_split_seed_action_audit_analysis_20261105_codex import load, summarize


def main():
    root = Path(__file__).resolve().parents[1] / "results"
    summary = {}
    for seed in (42, 1042, 2042):
        name = f"pc_bounded_branch_cem_task42_solver{seed}_20261105_codex"
        base = f"pc_bounded_cem_task42_solver{seed}_20261105_codex"
        traces = load(root / f"{name}.trace.jsonl")
        baseline = load(root / f"{base}.trace.jsonl")
        if len(traces) != 50 or traces.keys() != baseline.keys():
            raise ValueError("different task/reset identity")
        if any(not np.array_equal(traces[key][0]["before"], baseline[key][0]["before"])
               for key in traces):
            raise ValueError("physical snapshot mismatch")
        action_summary = summarize(traces)
        if action_summary["out_of_bounds_actions"]:
            raise ValueError("executed branch action exceeded physical box")
        audits = [json.loads(line) for line in (root / f"{name}.audit.jsonl").read_text().splitlines()]
        records = [row for audit in audits for row in audit["environments"]]
        if any(row["scored"] != 3000 for row in records):
            raise ValueError("invalid per-plan query accounting")
        new = json.loads((root / name / "eval_metrics.json").read_text())["episode_successes"]
        old = json.loads((root / base / "eval_metrics.json").read_text())["episode_successes"]
        if len(new) != 50 or len(old) != 50:
            raise ValueError("incomplete task metrics")
        summary[seed] = {
            **action_summary, "initial_snapshot_parity": True, "plan_calls": len(records),
            "projected": sum(row["projected"] for row in records),
            "fallback_split": sum(row["fallback_split"] for row in records),
            "branch_success": sum(new), "single_success": sum(old),
            "branch_only": sum(n and not o for n, o in zip(new, old)),
            "single_only": sum(o and not n for n, o in zip(new, old)),
        }
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
