"""Analyze predeclared constrained-action runs against matching native traces."""
import json
from pathlib import Path

import numpy as np

from experiments.pc_split_seed_action_audit_analysis_20261105_codex import load, summarize


def main():
    root = Path(__file__).resolve().parents[1] / "results"
    output = {}
    for seed in (42, 1042, 2042):
        name = f"pc_bounded_cem_task42_solver{seed}_20261105_codex"
        native = f"pc_split_seed_action_audit_task42_solver{seed}_20261105_codex"
        traces = load(root / f"{name}.trace.jsonl")
        native_traces = load(root / f"{native}.jsonl")
        if len(traces) != 50 or traces.keys() != native_traces.keys():
            raise ValueError("unmatched task or reset identities")
        if any(not np.array_equal(traces[key][0]["before"], native_traces[key][0]["before"])
               for key in traces):
            raise ValueError("initial physical snapshot mismatch")
        summary = summarize(traces)
        if summary["out_of_bounds_actions"]:
            raise ValueError(f"bounded execution violated action domain: {summary}")
        audit = [json.loads(line) for line in (root / f"{name}.audit.jsonl").read_text().splitlines()]
        projected = sum(sum(row["projected_candidates"]) for rec in audit for row in rec["iterations"])
        scored = sum((row["batch_end"] - row["batch_start"]) * row["scored_per_environment"]
                     for rec in audit for row in rec["iterations"])
        metrics = json.loads((root / name / "eval_metrics.json").read_text())
        native_metrics = json.loads((root / native / "eval_metrics.json").read_text())
        output[seed] = {**summary, "scored": scored, "projected": projected,
                        "projected_fraction": projected / scored,
                        "bounded_success": int(sum(metrics["episode_successes"])),
                        "native_success": int(sum(native_metrics["episode_successes"])),
                        "initial_snapshot_parity": True}
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
