"""Check new chain logging against the original seed-42 experiment."""
import json
from pathlib import Path

import numpy as np

from experiments.pc_split_seed_action_audit_analysis_20261105_codex import load


def main():
    root = Path(__file__).resolve().parents[1] / "results"
    for mode in ("lookahead", "effect"):
        stem = f"pc_branch_effect_{mode}_task42_solver42_20261105_"
        old, new = (stem + suffix for suffix in ("codex", "chaintrace"))
        old_metrics = json.loads((root / old / "eval_metrics.json").read_text())
        new_metrics = json.loads((root / new / "eval_metrics.json").read_text())
        old_trace = load(root / f"{old}.trace.jsonl")
        new_trace = load(root / f"{new}.trace.jsonl")
        if old_metrics != new_metrics or old_trace.keys() != new_trace.keys():
            raise ValueError("logging rerun differs in metrics or task identity")
        for key in old_trace:
            if len(old_trace[key]) != len(new_trace[key]):
                raise ValueError("logging rerun differs in executed length")
            for a, b in zip(old_trace[key], new_trace[key]):
                if not np.array_equal(a["action"], b["action"]) or not np.array_equal(a["before"], b["before"]):
                    raise ValueError("logging rerun differs in executed actions or states")
        audit = [json.loads(line) for line in (root / f"{new}.audit.jsonl").read_text().splitlines()]
        plans = [env for row in audit for env in row["environments"]]
        if not plans or any(not np.isfinite(np.asarray(p["selected_normalized_chain"])).all() for p in plans):
            raise ValueError("missing or nonfinite selected chain")
        print(json.dumps({"mode": mode, "tasks": len(new_trace), "plans_with_chain": len(plans),
                          "metrics_and_executed_trace_equal": True}))


if __name__ == "__main__":
    main()
