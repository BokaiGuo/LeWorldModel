"""Validate saved goal translation bases, scores and source-bound pair ranks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import reacher_goal_translation_subspace_20261102 as run


def sign(value):
    if not np.isfinite(value):
        raise ValueError("nonfinite score")
    return int(value > 0) - int(value < 0)


def validate(plan, report, original, digest):
    need = run.host.require
    need(report["schema"] == "reacher_goal_translation_subspace_result_v1"
         and report["plan_sha256"] == digest and report["P1_admission"] is False
         and report["model_before"] == report["model_after"] == plan["model_hash"]
         and report["goal_encodes"] == 10 and len(report["bases"]) == 2
         and len(report["rows"]) == len(original) == 26, "report identity")
    bases = {}
    for row in report["bases"]:
        seed = row["seed"]
        need(seed in (20263160, 20263161) and seed not in bases, "basis root")
        target = np.asarray(row["target"], np.float64)
        shifted = np.asarray(row["shifted"], np.float64)
        basis = np.asarray(row["basis"], np.float64)
        singular = np.asarray(row["singular"], np.float64)
        need(target.shape == (192,) and shifted.shape == (4, 192) and basis.shape == (4, 192)
             and singular.shape == (4,) and np.isfinite(shifted).all() and np.isfinite(basis).all()
             and np.isfinite(target).all() and np.isfinite(singular).all(), "basis shape/finite")
        # Original subtraction occurs on float32 tensors before SVD.
        differences = (shifted.astype(np.float32) - target.astype(np.float32)).astype(np.float64)
        expected = np.linalg.svd(differences, compute_uv=False)
        need(np.allclose(singular, expected, rtol=1e-9, atol=1e-10)
             and singular[-1] >= singular[0]*plan["relative_singular_minimum"]
             and np.allclose(basis @ basis.T, np.eye(4), rtol=1e-9, atol=1e-9)
             and np.allclose((differences @ basis.T) @ basis, differences, rtol=1e-8, atol=1e-8),
             "invalid SVD basis")
        bases[seed] = (target, basis)
    counts = {}
    for row, source in zip(report["rows"], original, strict=True):
        seed, prefix = row["seed"], row["prefix"]
        need((seed, prefix) == (source["seed"], source["prefix"])
             and row["actions_sha256"] == source["actions_sha256"]
             and row["final_rgb_sha256"] == source["final_rgb_sha256"]
             and row["physical_final"] == source["physical_final"], "source identity")
        target, basis = bases[seed]
        need(np.allclose(target, source["target"], atol=1e-6, rtol=0), "goal binding")
        scores = []
        for key, field in (("predicted", "model_score"), ("actual", "real_score")):
            coord = np.asarray(source[key], np.float64)
            need(coord.shape == (2, 192) and np.isfinite(coord).all(), "coordinate shape")
            calculated = np.mean(((coord-target) @ basis.T)**2, axis=1)
            need(np.allclose(calculated, row[field], rtol=1e-10, atol=1e-12), "score mismatch")
            scores.append(sign(calculated[1]-calculated[0]))
        physical = sign(row["physical_final"][1]-row["physical_final"][0])
        group = counts.setdefault(str(seed), {"pairs": 0, "model_real_reversals": 0,
            "real_physical_reversals": 0, "model_physical_reversals": 0, "model_ties": 0,
            "real_ties": 0, "physical_ties": 0})
        model, actual = scores
        group["pairs"] += 1
        group["model_real_reversals"] += model*actual < 0
        group["real_physical_reversals"] += actual*physical < 0
        group["model_physical_reversals"] += model*physical < 0
        group["model_ties"] += model == 0
        group["real_ties"] += actual == 0
        group["physical_ties"] += physical == 0
    need(set(counts) == {"20263160", "20263161"} and all(x["pairs"] == 13 for x in counts.values()),
         "root roster")
    return {"schema": "reacher_goal_translation_subspace_saved_v1",
            "status": "saved_consistency_complete", "roots": counts, "P1_admission": False,
            "limits": ["exposed roots and dependent pairs", "basis uses full-image shifts including background",
                       "saved encoder output not independently rerun", "no physical planner evaluation"]}


def main():
    p = argparse.ArgumentParser()
    for arg in ("plan", "plan-sha", "report", "report-sha", "output"):
        p.add_argument("--" + arg, required=True)
    args = p.parse_args()
    plan_path, report_path = Path(args.plan), Path(args.report)
    need = run.host.require
    need(run.setup._sha256_file(plan_path) == args.plan_sha
         and run.setup._sha256_file(report_path) == args.report_sha, "input hashes")
    plan = json.loads(plan_path.read_text())
    need(plan == run.prepare(), "source drift")
    result = validate(plan, json.loads(report_path.read_text()), json.loads(run.REPORT.read_text())["rows"],
                      args.plan_sha)
    result.update(plan_sha256=args.plan_sha, report_sha256=args.report_sha,
                  verifier_sha256=run.setup._sha256_file(Path(__file__)))
    run.host.write_new(args.output, result)
    print(json.dumps(result["roots"], sort_keys=True))


if __name__ == "__main__":
    main()
