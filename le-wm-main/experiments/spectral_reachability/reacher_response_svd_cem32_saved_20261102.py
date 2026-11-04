"""Saved-only recomputation of the full-bank CEM32 selection."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import reacher_response_svd_cem32_20261102 as run


def validate(plan, report, source):
    require = run.host.require
    require(report["status"] == "complete" and report["model_before"] == report["model_after"] == plan["model_hash"]
            and [x["seed"] for x in report["rows"]] == [20263160, 20263161], "completion/model/roots")
    output = []
    for row, bank in zip(report["rows"], source["rows"], strict=True):
        require(row["seed"] == bank["seed"] and len(bank["bank_params"]) == 13, "source bank")
        details, candidates = row["selection"], row["selection"]["proposal_rows"]
        require(len(candidates) == 13 and [x["proposal_index"] for x in candidates] == list(range(13))
                and details["cem_seed_policy"] == "same seed at every candidate"
                and details["suffix_cem_calls"] == 13 and details["suffix_cem_rollout_evaluations"] == 1248
                and details["predict_sample_evaluations"] == 2522, "roster/cost")
        for candidate in candidates:
            receipt = candidate["cem_receipt"]
            require(all(math.isfinite(candidate[k]) for k in ("immediate_goal_cost", "cem_future_goal_cost"))
                    and receipt["forward_rollout_evaluations"] == 96
                    and receipt["model_predict_sample_evaluations"] == 192
                    and receipt["best_cost"] == candidate["cem_future_goal_cost"]
                    and len(candidate["suffix_parameters"]) == 2
                    and all(len(x) == 10 and all(math.isfinite(v) for v in x)
                            for x in candidate["suffix_parameters"]), "candidate receipt")
        reference = candidates[0]
        eligible = [c for c in candidates[1:]
                    if c["immediate_goal_cost"] >= reference["immediate_goal_cost"] + plan["non_greedy_margin"]
                    and c["cem_future_goal_cost"] <= reference["cem_future_goal_cost"] - plan["future_margin"]]
        chosen = min(eligible, key=lambda x: (x["cem_future_goal_cost"], x["proposal_index"])) if eligible else None
        selected = chosen["proposal_index"] if chosen else None
        greedy = min(candidates, key=lambda x: (x["immediate_goal_cost"], x["proposal_index"]))["proposal_index"]
        require(selected == details["selected_proposal_index"] and greedy == row["same_bank_greedy"]
                and len(eligible) == details["eligible_count"]
                and row["selected_params"] == (None if chosen is None else [bank["bank_params"][selected]])
                and details["selected_suffix_parameters"] == (None if chosen is None else chosen["suffix_parameters"]),
                "selector mismatch")
        output.append({"seed": row["seed"], "selected": selected, "neutral": 0, "same_bank_greedy": greedy,
                       "eligible_count": len(eligible), "predict_samples": details["predict_sample_evaluations"]})
    return {"status": "saved_selection_recomputed", "rows": output, "P1_admission": False,
            "independent_confirmation": False, "gpu_scores_independently_recomputed": False,
            "scope": "known roots; model-only ranking; physical outcomes not used as selector inputs"}


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan", "plan-sha", "report", "report-sha", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    for path, digest in ((args.plan, args.plan_sha), (args.report, args.report_sha),
                         (run.SOURCE, run.SOURCE_SHA)):
        run.host.require(run.setup._sha256_file(Path(path)) == digest, "input hash")
    plan, report, source = [json.loads(Path(x).read_text()) for x in (args.plan, args.report, run.SOURCE)]
    run.host.require(plan == run.make_plan() and report["plan_sha256"] == args.plan_sha, "plan binding")
    result = validate(plan, report, source)
    result.update(plan_sha256=args.plan_sha, report_sha256=args.report_sha,
                  consumer_sha256=run.setup._sha256_file(Path(__file__)))
    run.host.write_new(args.output, result)
    print(json.dumps({"rows": result["rows"], "sha256": run.setup._sha256_file(Path(args.output))}))


if __name__ == "__main__":
    main()
