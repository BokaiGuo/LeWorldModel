"""Validate the sealed model-only sensitivity journal and original-seed replay."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import reacher_cem_context_sensitivity_v2_20261102 as run


def validate(plan, events, plan_sha):
    require = run.host.require
    require(events[0]["kind"] == "start" and events[0]["plan_sha256"] == plan_sha
            and events[-1]["kind"] == "finished"
            and events[-1]["model_before"] == events[-1]["model_after"] == plan["model_parameter_hash"],
            "run framing or model")
    expected = [(root, prefix, budget, root + 5000 + offset)
                for root in plan["roots"] for prefix in plan["prefixes"]
                for budget in plan["budgets"] for offset in plan["seed_offsets"]]
    rows = events[1:-1]
    require(len(rows) == len(expected) == 24, "pair count")
    original = json.loads(run.physical.REPORT.read_text())
    require(run.setup._sha256_file(run.physical.REPORT) == run.physical.REPORT_SHA, "original report hash")
    previous = {r["seed"]: r for r in original["rows"]}
    counts = {"16x2": 0, "32x3": 0}
    reproduced = 0
    for row, (root, prefix, budget, seed) in zip(rows, expected, strict=True):
        require(row["kind"] == "pair" and (row["seed"], row["prefix"], row["budget"], row["cem_seed"])
                == (root, prefix, budget, seed), "pair order")
        require(math.isfinite(row["context_mse"]) and row["context_mse"] >= 0
                and 0 <= row["history_max_abs_error"] < 1e-5, "context or history")
        suffix = []
        for label in plan["contexts"]:
            item = row["contexts"][label]
            params = np.asarray(item["suffix_parameters"], np.float32)
            receipt = item["receipt"]
            require(params.shape == (2, 10) and np.isfinite(params).all()
                    and math.isfinite(item["wall_seconds"]) and item["wall_seconds"] >= 0
                    and math.isfinite(receipt["best_cost"]) and len(receipt["trace"]) == budget[1]
                    and all(math.isfinite(x) for x in receipt["trace"])
                    and receipt["forward_rollout_evaluations"] == budget[0] * budget[1]
                    and receipt["model_predict_sample_evaluations"] == budget[0] * budget[1] * 2,
                    "CEM result contract")
            suffix.append(params)
        delta = float(np.max(np.abs(suffix[0] - suffix[1])))
        require(abs(delta - row["parameter_max_abs_difference"]) < 1e-6, "difference mismatch")
        if delta > 1e-5:
            counts["16x2" if budget[0] == 16 else "32x3"] += 1
        if budget == [16, 2, 4] and seed == root + 5000:
            prior = previous[root]["selection"]["proposal_rows"][prefix]
            require(prior["proposal_index"] == prefix
                    and np.array_equal(suffix[0], np.asarray(prior["suffix_parameters"], np.float32))
                    and row["contexts"]["predicted"]["receipt"]["best_cost"] == prior["cem_future_goal_cost"],
                    "original predicted CEM replay mismatch")
            reproduced += 1
    require(reproduced == 4, "replay denominator")
    return {"status": "saved_consistency", "pairs": len(rows), "original_predicted_reproduced": reproduced,
            "material_parameter_difference_threshold": 1e-5, "material_differences": counts,
            "scope": "two exposed roots; no new physical actions; no independent simulator or GPU replay",
            "independent_confirmation": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan", "plan-sha256", "journal", "journal-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    plan_path, journal_path = Path(args.plan), Path(args.journal)
    require = run.host.require
    require(run.setup._sha256_file(plan_path) == args.plan_sha256
            and run.setup._sha256_file(journal_path) == args.journal_sha256, "input hashes")
    plan = json.loads(plan_path.read_text())
    require(plan == run.prepare(), "plan/source drift")
    result = validate(plan, run.saved.verify_journal(journal_path), args.plan_sha256)
    result.update(plan_sha256=args.plan_sha256, journal_sha256=args.journal_sha256,
                  consumer_sha256=run.setup._sha256_file(Path(__file__)))
    run.host.write_new(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
