"""Verify robust rescoring and summarize CEM action-bank coverage."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_robust_coverage_20261102 as source
from . import p1_dev305_compass_model_compare_saved_20261102 as nominal

base, need = source.base, source.need
ROOT = source.ROOT / "results"
CONTRACT_SHA = "f2a4beee8502dc86e1d217e87b4d87e64fcf8983a6c913885ee8722ad2eb94db"
REPORT_SHA = "3e53492c59347ef0e6cbcb649e8fbeaca854f97630e4be70c0f0b5fe785f16f5"


def analyze():
    contract = base.read(ROOT / f"{source.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    report = base.read(ROOT / f"{source.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    old = base.read(ROOT / f"{source.planner.NAME}_PLAN_DEV01/report.json", source.physical.source.PLAN_SHA)
    previous = base.read(ROOT / f"{nominal.source.NAME}_RUN_DEV01/report.json", nominal.REPORT_SHA)
    need(contract == source.protocol() and report["contract_sha256"] == CONTRACT_SHA
         and report["model_before"] == report["model_after"] == contract["model_hash"], "source")
    common = source.run.coherent.base.first.engine.base.common
    need(len(report["rows"]) == 54, "roster")
    results = []
    for role in contract["roles"]:
        saved = [r for r in report["rows"] if r["arm"] == role["arm"]]
        original = next(r for r in old["rows"] if r["arm"] == role["arm"])
        need(len(saved) == len(role["rows"]) == 18 and len(original["iterations"]) == 5, "arm roster")
        banks = np.concatenate([np.asarray(it["bank"], dtype=np.float32) for it in original["iterations"]])
        need(banks.shape == (320, 2, 5, 2), "bank shape")
        feasible = np.concatenate([np.asarray(it["feasible"], dtype=bool) for it in original["iterations"]])
        for item, row in zip(role["rows"], saved, strict=True):
            scores = np.asarray(row["variant_goal_mse"], dtype=np.float64)
            old_nominal = next(r for r in previous["rows"] if r["arm"] == role["arm"]
                               and r["name"] == item["name"])
            need(row["arm"] == role["arm"] and row["name"] == item["name"]
                 and row["actions_sha256"] == item["actions_sha256"] and row["feasible"] == item["feasible"]
                 and scores.shape == (5,) and np.isfinite(scores).all()
                 and np.isclose(row["robust_score"], common.robust_scores(scores[None])[0], atol=1e-12)
                 and np.isclose(scores[0], old_nominal["predicted_goal_mse"], atol=2e-6), "score binding")
            action = np.asarray(item["actions"], dtype=np.float32).reshape(2, 5, 2)
            distances = np.sqrt(np.mean((banks-action)**2, axis=(1, 2, 3)))
            results.append({"arm": role["arm"], "name": item["name"], "feasible": row["feasible"],
                            "robust_score": row["robust_score"], "nearest_bank_rms": float(distances.min()),
                            "nearest_feasible_bank_rms": float(distances[feasible].min()),
                            "exact_sampled": bool((distances == 0).any())})
        selected = next(r for r in saved if r["name"] == "cem_selected")
        need(np.isclose(selected["robust_score"], original["selected"]["robust_score"], atol=2e-6),
             "original robust score")
    return {"schema": source.NAME + ".analysis.v1", "rows": results,
            "scope": "post-outcome fixed development bank; Euclidean raw-action RMS is not physical reachability",
            "independent_confirmation": False, "p1_admission": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
