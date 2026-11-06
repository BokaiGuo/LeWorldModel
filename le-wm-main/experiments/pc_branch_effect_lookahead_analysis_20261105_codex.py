"""Paired analysis with physical identity and budget checks; prints JSON only."""
import json
from pathlib import Path

import numpy as np

from experiments.pc_split_seed_action_audit_analysis_20261105_codex import load, summarize


def main():
    root = Path(__file__).resolve().parents[1] / "results"
    summary = {}
    for seed in (42, 1042, 2042):
        arms = {}
        for mode in ("lookahead", "effect"):
            name = f"pc_branch_effect_{mode}_task42_solver{seed}_20261105_codex"
            traces = load(root / f"{name}.trace.jsonl")
            metrics = json.loads((root / name / "eval_metrics.json").read_text())
            success = metrics["episode_successes"]
            audit = [json.loads(line) for line in (root / f"{name}.audit.jsonl").read_text().splitlines()]
            plans = [x for record in audit for x in record["environments"]]
            action = summarize(traces)
            if len(traces) != 50 or len(success) != 50 or action["out_of_bounds_actions"]:
                raise ValueError("incomplete panel or illegal action")
            if not plans or any(x["scored"] != 3000 or len(x["rounds"]) != 9 for x in plans):
                raise ValueError("incorrect query budget")
            rounds = [r for p in plans for r in p["rounds"]]
            arms[mode] = {"traces": traces, "success": success, "plan_calls": len(plans),
                          "scored": 3000 * len(plans), "actions": action,
                          "proposal_elites": sum(sum(r["proposal_elites"]) for r in rounds),
                          "distinct_effect_slots": sum(v > 1e-6 for r in rounds
                                                       for branch in r["proposal_diff_from_random"] for v in branch)}
        a, b = arms["lookahead"], arms["effect"]
        if a["traces"].keys() != b["traces"].keys() or any(
            not np.array_equal(a["traces"][k][0]["before"], b["traces"][k][0]["before"])
            for k in a["traces"]
        ):
            raise ValueError("physical initial conditions differ")
        summary[seed] = {
            "initial_state_equal": True,
            "lookahead_success": sum(a["success"]), "effect_success": sum(b["success"]),
            "lookahead_only": sum(x and not y for x, y in zip(a["success"], b["success"])),
            "effect_only": sum(y and not x for x, y in zip(a["success"], b["success"])),
            "lookahead_plans": a["plan_calls"], "effect_plans": b["plan_calls"],
            "lookahead_scored": a["scored"], "effect_scored": b["scored"],
            "lookahead_proposal_elites": a["proposal_elites"], "effect_proposal_elites": b["proposal_elites"],
            "effect_distinct_slots": b["distinct_effect_slots"],
            "lookahead_actions": a["actions"], "effect_actions": b["actions"],
        }
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
