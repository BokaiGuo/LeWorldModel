"""Recompute saved goal-Q scores and bind each pair to its source evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import reacher_fixed_pair_goal_q_20261102 as run

RIDGE = 0.01


def score(gram, value, target):
    """Independent linear solve, equivalent to the existing squared Q metric."""
    gram = np.asarray(gram, np.float64)
    value, target = np.asarray(value, np.float32), np.asarray(target, np.float32)
    if (gram.ndim != 2 or gram.shape[0] != gram.shape[1] or target.shape != (gram.shape[0],)
            or value.shape != (2, gram.shape[0]) or not np.isfinite(gram).all()
            or not np.isfinite(value).all() or not np.isfinite(target).all()
            or not np.allclose(gram, gram.T, atol=1e-12, rtol=1e-12)):
        raise ValueError("invalid Q score inputs")
    # q_costs subtracts model float32 coordinates before promoting to float64.
    diff = (value - target).astype(np.float64)
    return RIDGE * np.einsum("bi,ib->b", diff, np.linalg.solve(
        gram + RIDGE * np.eye(len(target)), diff.T)) / len(target)


def sign(x):
    if not np.isfinite(x):
        raise ValueError("nonfinite rank")
    return int(x > 0) - int(x < 0)


def validate(plan, report, replay, original, grams, digest):
    need = run.host.require
    need(report["schema"] == "reacher_fixed_pair_goal_q_result_v1"
         and report["plan_sha256"] == digest and report["P1_admission"] is False
         and report["model_before"] == report["model_after"] == plan["model_hash"], "report binding")
    need(len(report["rows"]) == len(original) == 26 and len(replay) == 52, "denominator")
    counts, max_residual, worst_model_l2 = {}, 0., 0.
    for i, (row, source) in enumerate(zip(report["rows"], original, strict=True)):
        seed, prefix = row["seed"], row["prefix"]
        need((seed, prefix) == (source["seed"], source["prefix"])
             and row["cp5_rgb_sha256"] == source["cp5_rgb_sha256"], "source identity")
        expected = replay[2*i:2*i+2]
        for j, item in enumerate(expected):
            need((item["seed"], item["prefix"], item["label"])
                 == (seed, prefix, ("l2", "direction")[j])
                 and item["actions_sha256"] == row["actions_sha256"][j]
                 and item["final_rgb_sha256"] == row["final_rgb_sha256"][j]
                 and item["final_distance"] == row["physical_final"][j], "replay binding")
        target = np.asarray(row["target"], np.float64)
        pred = np.asarray(row["predicted"], np.float64)
        actual = np.asarray(row["actual"], np.float64)
        pred_q, actual_q = score(grams[seed], pred, target), score(grams[seed], actual, target)
        for computed, stored in ((pred_q, row["predicted_q"]), (actual_q, row["actual_q"])):
            residual = np.max(np.abs(computed - np.asarray(stored)))
            max_residual = max(max_residual, float(residual))
            need(np.allclose(computed, stored, rtol=1e-8, atol=1e-10), "Q score mismatch")
        l2 = np.mean((pred-target)**2, axis=1)
        worst_model_l2 = max(worst_model_l2, float(np.max(np.abs(l2 - source["l2_model_cost"]))))
        need(np.allclose(l2, source["l2_model_cost"], rtol=1e-5, atol=1e-7), "prediction L2 mismatch")
        group = counts.setdefault(str(seed), {"pairs": 0, "model_real_reversals": 0,
                                               "real_physical_reversals": 0,
                                               "model_physical_reversals": 0,
                                               "model_ties": 0, "real_ties": 0, "physical_ties": 0})
        model = sign(pred_q[1]-pred_q[0])
        observed = sign(actual_q[1]-actual_q[0])
        physical = sign(row["physical_final"][1]-row["physical_final"][0])
        group["pairs"] += 1
        group["model_real_reversals"] += model*observed < 0
        group["real_physical_reversals"] += observed*physical < 0
        group["model_physical_reversals"] += model*physical < 0
        group["model_ties"] += model == 0
        group["real_ties"] += observed == 0
        group["physical_ties"] += physical == 0
    need(set(counts) == {"20263160", "20263161"} and all(v["pairs"] == 13 for v in counts.values()),
         "root roster")
    return {"schema": "reacher_fixed_pair_goal_q_saved_v1", "status": "saved_consistency_complete",
            "roots": counts, "max_q_recomputation_abs_error": max_residual,
            "max_model_l2_recomputation_abs_error": worst_model_l2,
            "P1_admission": False, "limits": ["exposed roots; dependent pairs",
                "saved latent vectors checked but neural encodings not independently recomputed",
                "real outcome scores are post-execution, not deployable imagined-image scores"]}


def main():
    p = argparse.ArgumentParser()
    for arg in ("plan", "plan-sha", "report", "report-sha", "output"):
        p.add_argument("--" + arg, required=True)
    args = p.parse_args()
    plan_path, report_path = Path(args.plan), Path(args.report)
    need = run.host.require
    need(run.setup._sha256_file(plan_path) == args.plan_sha
         and run.setup._sha256_file(report_path) == args.report_sha, "input hash")
    plan = json.loads(plan_path.read_text())
    need(plan == run.prepare(), "plan/source drift")
    replay = json.loads(run.old.RECEIPT.read_text())["rows"]
    original = json.loads(run.old.PRED.read_text())["rows"]
    grams = {r["seed"]: r["goal_metric"]["gram"] for r in json.loads(run.preflight.Q_PATH.read_text())["rows"]}
    result = validate(plan, json.loads(report_path.read_text()), replay, original, grams, args.plan_sha)
    result.update(plan_sha256=args.plan_sha, report_sha256=args.report_sha,
                  verifier_sha256=run.setup._sha256_file(Path(__file__)))
    run.host.write_new(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
