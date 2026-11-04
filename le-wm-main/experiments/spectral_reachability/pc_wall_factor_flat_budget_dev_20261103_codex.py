"""Equal full-rollout-count flat CEM development control."""
import json
from pathlib import Path

import numpy as np

from . import pc_wall_retention_factor_dev_20261103_codex as factor

base = factor.prior.base
PROTOCOL = factor.ROOT.parent / "docs/PC_WALL_FACTOR_FLAT_BUDGET_DEV_PROTOCOL_20261103_CODEX.md"
OUTPUT = factor.ROOT / "results/pc_wall_factor_flat_budget_dev_20261103_codex.json"
BUDGETS = ((128, 1145), (256, 2169))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    original = factor.load_prior()
    report = json.loads(factor.OUTPUT.read_text())
    seal = json.loads(factor.SEAL.read_text())
    if report["seal_sha256"] != base.digest(factor.SEAL) or seal["prior_sha256"] != base.digest(factor.prior.OUTPUT):
        raise ValueError("factor provenance mismatch")
    paths = (PROTOCOL, Path(__file__), factor.OUTPUT, factor.SEAL, factor.prior.OUTPUT,
             Path(factor.__file__), Path(base.__file__), Path(base.physics.__file__))
    hashes = {str(p): base.digest(p) for p in paths}
    rows = []
    print("Exposed exact tasks: equal rollout counts, not equal total hardware cost.", flush=True)
    for row, prior_row in zip(report["rows"], original["rows"], strict=True):
        task = prior_row["task"]
        for suffix_cap, cap in BUDGETS:
            recorded = next(x for x in row["budgets"] if x["cap"] == suffix_cap)
            flat = base.plan(task, row["index"], cap, "flat")
            methods = {}
            for name in ("diverse_original", "diverse_stratified"):
                arm = recorded["arms"][name]
                if arm["screening_calls"] + arm["continuation_calls"] != cap:
                    raise ValueError("rollout budget mismatch")
                methods[name] = {"distance": arm["terminal_distance"], "actions": arm["actions"],
                                 "success": arm["terminal_distance"] < .05}
            rows.append({"task": row["index"], "suffix_cap": suffix_cap, "rollout_budget": cap,
                         "flat": flat, "programs": methods})
        print(f"completed task {row['index']}", flush=True)
    if any(base.digest(Path(p)) != d for p, d in hashes.items()):
        raise ValueError("dependency drift")
    summary = {}
    for suffix_cap, cap in BUDGETS:
        subset = [r for r in rows if r["suffix_cap"] == suffix_cap]
        flat = [r["flat"]["terminal_distance"] for r in subset]
        summary[str(cap)] = {"flat_mean": float(np.mean(flat)),
                             "flat_successes": sum(x < .05 for x in flat)}
        for arm in ("diverse_original", "diverse_stratified"):
            vals = [r["programs"][arm]["distance"] for r in subset]
            summary[str(cap)][arm] = {"mean": float(np.mean(vals)), "successes": sum(x < .05 for x in vals),
                                      "wins_vs_flat": sum(x < y for x, y in zip(vals, flat, strict=True))}
    with OUTPUT.open("x") as handle:
        json.dump({"dependencies": hashes, "rows": rows, "summary": summary,
                   "scope": "exposed exact development, equal rollout count only",
                   "PC3_pass": False, "full_cost_certified": False}, handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
