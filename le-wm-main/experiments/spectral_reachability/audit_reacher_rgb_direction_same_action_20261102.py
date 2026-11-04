"""Saved-only source binding and rank summary for same-action re-scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import reacher_rgb_direction_same_action_20261102 as run


def validate(plan, report, source_events, plan_sha):
    need = run.host.require
    need(report["schema"] == "reacher_rgb_direction_same_action_result_v1"
         and report["plan_sha256"] == plan_sha and report["model_before"] == report["model_after"] == plan["model_hash"]
         and report["predict_calls"] == 52 and report["predict_samples"] == 104
         and len(report["rows"]) == 26, "report framing")
    source_plan = json.loads(run.PLAN.read_text())
    mean, std = run.setup._load_action_stats(Path(source_plan["stats"]), "swm/ReacherDMControl-v0")
    ctx = run.paired.audit.context_lib
    out, cursor = [], 0
    for root in source_plan["roots"]:
        summary = {"seed": root["seed"], "pairs": 0, "physical_direction_better": 0,
                   "l2_model_direction_better": 0, "direction_model_direction_better": 0,
                   "l2_reversed": 0, "direction_reversed": 0, "ties": 0}
        for arm in root["arms"]:
            row = report["rows"][cursor]
            cursor += 1
            outcomes, contexts = [], []
            for events in source_events:
                matches = [e for e in events if e.get("seed") == root["seed"] and e.get("arm") == arm["arm"]
                           and e["kind"] == "arm_returned"]
                context = [e for e in events if e.get("seed") == root["seed"] and e.get("arm") == arm["arm"]
                           and e["kind"] == "context" and e["role"] == "successor"]
                need(len(matches) == len(context) == 1, "source arm")
                outcomes.append(matches[0]["result"])
                contexts.append(context[0]["value"])
            actions = np.asarray([r["action_witness_normalized"][1:] for r in outcomes], np.float32)
            residuals = []
            for action, outcome in zip(actions, outcomes, strict=True):
                raw = np.asarray(outcome["action_witness_raw"], np.float32)[5:]
                rebuilt = run.setup.p0._normalized_blocks_to_raw_actions(action, mean, std, action_dim=2)
                residuals.append(float(np.max(np.abs(rebuilt - raw))))
            physical = [r["physical"]["final_qpos_distance"] for r in outcomes]
            frame = ctx.unpack_rgb_observation(contexts[0]["frames"][-1])
            goal = ctx.unpack_rgb_observation(contexts[0]["goal"])
            need(row["seed"] == root["seed"] and row["prefix"] == arm["proposal_index"]
                 and np.array_equal(np.asarray(row["normalized_actions"], np.float32), actions)
                 and row["physical_final"] == physical and row["cp5_rgb_sha256"] == ctx.rgb_digest(frame)
                 and row["goal_sha256"] == ctx.rgb_digest(goal)
                 and row["normalized_raw_max_abs"] == residuals, "result/source binding")
            l2, direction = np.asarray(row["l2_model_cost"]), np.asarray(row["direction_model_cost"])
            need(l2.shape == direction.shape == (2,) and np.isfinite(l2).all() and np.isfinite(direction).all()
                 and len(row["normalized_raw_max_abs"]) == 2
                 and all(0 <= x <= 1e-6 for x in row["normalized_raw_max_abs"]), "cost or reconstruction")
            pd, ld, dd = physical[1] - physical[0], float(l2[1] - l2[0]), float(direction[1] - direction[0])
            summary["pairs"] += 1
            summary["physical_direction_better"] += int(pd < 0)
            summary["l2_model_direction_better"] += int(ld < 0)
            summary["direction_model_direction_better"] += int(dd < 0)
            summary["l2_reversed"] += int(pd * ld < 0)
            summary["direction_reversed"] += int(pd * dd < 0)
            summary["ties"] += int(pd == 0 or ld == 0 or dd == 0)
        out.append(summary)
    need(cursor == 26, "denominator")
    return {"status": "saved_source_binding_complete", "roots": out, "P1_admission": False,
            "limitations": ["saved scores not independently recomputed from prediction vectors",
                            "two exposed roots; pairs share context and are not independent",
                            "model-latent vs physical ranking mixes prediction and task metric effects"]}


def main():
    parser = argparse.ArgumentParser()
    for key in ("plan", "plan-sha", "report", "report-sha", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    need = run.host.require
    need(run.setup._sha256_file(Path(args.plan)) == args.plan_sha
         and run.setup._sha256_file(Path(args.report)) == args.report_sha, "external hashes")
    plan = json.loads(Path(args.plan).read_text())
    need(plan == run.prepare(), "plan/source drift")
    report = json.loads(Path(args.report).read_text())
    events = [run.paired.audit.saved.verify_journal(p) for p in (run.paired.prior.L2_JOURNAL, run.JOURNAL)]
    result = validate(plan, report, events, args.plan_sha)
    result.update(plan_sha256=args.plan_sha, report_sha256=args.report_sha,
                  consumer_sha256=run.setup._sha256_file(Path(__file__)))
    run.host.write_new(args.output, result)
    print(json.dumps({"sha256": run.setup._sha256_file(Path(args.output)), "roots": result["roots"]}))


if __name__ == "__main__":
    main()
