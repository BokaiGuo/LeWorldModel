"""Validate frozen selection report and attach previously opened physical outcomes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import reacher_response_svd_predicted_cem_20261102 as run

host = run.host
PHYSICAL = run.ROOT / "results/reacher_pc_response_svd_physical_SAVED_20261102_v1.json"
PHYSICAL_SHA = "c3e77d943e65d2bf7de5ba9f32f7fc5fb9299ad86626e5567b074ab1d1f8399f"
FEEDBACK = run.ROOT / "results/reacher_response_svd_cp5_SAVED_20261102_v1.json"
FEEDBACK_SHA = "20c412e0bfd3d7cc0710005af62892ec9ab805b76087c988917e2917b9f44207"


def validate(report, plan, source, physical, feedback):
    host.require(report["status"] == "complete" and report["model_before"] == report["model_after"]
                 == plan["model_hash"] and [r["seed"] for r in report["rows"]] == [20263160, 20263161],
                 "run completion or model")
    output = []
    for record, bank_row in zip(report["rows"], source["rows"], strict=True):
        seed, details = record["seed"], record["selection"]
        host.require(seed == bank_row["seed"] and len(bank_row["bank_params"]) == 13, "source bank")
        rows = details["proposal_rows"]
        host.require(len(rows) == 13 and [r["proposal_index"] for r in rows] == list(range(13))
                     and details["cem_seed_policy"] == "same seed at every candidate"
                     and details["suffix_cem_calls"] == 13
                     and details["suffix_cem_rollout_evaluations"] == 416
                     and details["predict_sample_evaluations"] == 858, "selection roster or costs")
        for item in rows:
            host.require(all(math.isfinite(item[k]) for k in ("immediate_goal_cost", "cem_future_goal_cost"))
                         and item["cem_receipt"]["forward_rollout_evaluations"] == 32
                         and item["cem_receipt"]["model_predict_sample_evaluations"] == 64
                         and item["cem_receipt"]["best_cost"] == item["cem_future_goal_cost"]
                         and len(item["suffix_parameters"]) == 2
                         and all(len(p) == 10 and all(math.isfinite(x) for x in p)
                                 for p in item["suffix_parameters"]), "candidate fields")
        neutral = rows[0]
        eligible = [r for r in rows[1:]
                    if r["immediate_goal_cost"] >= neutral["immediate_goal_cost"] + 1e-6
                    and r["cem_future_goal_cost"] <= neutral["cem_future_goal_cost"] - 1e-6]
        winner = min(eligible, key=lambda x: (x["cem_future_goal_cost"], x["proposal_index"])) if eligible else None
        index = None if winner is None else winner["proposal_index"]
        host.require(index == details["selected_proposal_index"] and len(eligible) == details["eligible_count"]
                     and record["selected_params"] == (None if winner is None else [bank_row["bank_params"][index]])
                     and details["selected_suffix_parameters"] == (None if winner is None else winner["suffix_parameters"]),
                     "selection recomputation")
        item = {"seed": seed, "selected": index, "eligible_count": len(eligible)}
        if winner is not None:
            cp = [r for r in feedback["results"] if r["seed"] == seed and r["prefix"] in (0, index)]
            panel = [r for r in physical["paired_rows"] if r["seed"] == seed and r["prefix"] == index]
            host.require(len(cp) == 2 and {r["prefix"] for r in cp} == {0, index}
                         and len(panel) == 5 and sorted(r["tail"] for r in panel) == list(range(5))
                         and all(r["status"] == "complete" for r in panel), "prior physical roster")
            item.update(prior_feedback=cp, prior_common_tail=panel)
        output.append(item)
    return {"status": "saved_selection_recomputed", "rows": output,
            "scope": "model selection and previously opened development outcomes; not independent or P1",
            "latent_scores_not_independently_recomputed": True, "physical_outcomes_used_for_selection": False,
            "equal_total_compute": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan", "plan-sha", "report", "report-sha", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    for path, expected in ((args.plan, args.plan_sha), (args.report, args.report_sha),
                           (PHYSICAL, PHYSICAL_SHA), (FEEDBACK, FEEDBACK_SHA)):
        host.require(run.setup._sha256_file(Path(path)) == expected, "source hash")
    plan = json.loads(Path(args.plan).read_text())
    report = json.loads(Path(args.report).read_text())
    host.require(plan == run.make_plan() and report["plan_sha256"] == args.plan_sha, "sealed plan binding")
    result = validate(report, plan, json.loads(run.SOURCE.read_text()), json.loads(PHYSICAL.read_text()),
                      json.loads(FEEDBACK.read_text()))
    result.update(plan_sha256=args.plan_sha, report_sha256=args.report_sha,
                  physical_sha256=PHYSICAL_SHA, feedback_sha256=FEEDBACK_SHA)
    host.write_new(args.output, result)
    print(json.dumps({"status": result["status"], "sha256": run.setup._sha256_file(Path(args.output))}))


if __name__ == "__main__":
    main()
