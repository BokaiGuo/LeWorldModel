"""Recompute paired prediction errors and rankings from sealed CUDA arrays."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from . import p1_dev305_refresh_diagnostic_20261102 as run

SPEC_SHA = "0ad9513e09b7a37f53ac987a29ef76fe4ac9e101f20b01fb465c8493e8b3903d"
REPORT_SHA = "e620eca82882390e1c0a54a519daaefa149428a192b7408993a040839c5f90ac"


def analyze():
    root = run.ROOT/"results"
    spec = run.base.read(root/f"{run.NAME}_PREP_DEV01.json", SPEC_SHA)
    report = run.base.read(root/f"{run.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    run.need(spec == run.protocol() and report["contract_sha256"] == SPEC_SHA
             and report["model_before"] == report["model_after"] == spec["model_hash"], "source/model")
    run.need(report["counts"] == {"encoder": {"calls": 54, "samples": 54},
             "projector": {"calls": 54, "samples": 54}, "action_encoder": {"calls": 1, "samples": 18},
             "predictor": {"calls": 1, "samples": 18}}, "ledger")
    goal = np.asarray(spec["goal_latent"], float)
    rows = []
    for src, computed in zip(spec["rows"], report["rows"], strict=True):
        run.need(src["arm"] == computed["arm"], "roster")
        actual, opened, refreshed = (np.asarray(x, float) for x in
                                      (src["true_terminal"], src["open_terminal"], computed["refreshed_terminal"]))
        run.need(all(x.shape == goal.shape == (192,) and np.isfinite(x).all()
                     for x in (actual, opened, refreshed)), "finite latents")
        rows.append({"arm": src["arm"], "prefix": src["prefix"], "candidate": src["candidate"],
                     "open_mse": float(np.mean((opened-actual)**2)),
                     "refresh_mse": float(np.mean((refreshed-actual)**2)),
                     "open_goal": float(np.mean((opened-goal)**2)),
                     "refresh_goal": float(np.mean((refreshed-goal)**2)),
                     "real_goal": float(np.mean((actual-goal)**2))})
    counts = {"open": {"agree": 0, "reverse": 0, "tie": 0},
              "refresh": {"agree": 0, "reverse": 0, "tie": 0}}
    for prefix in ("original", "coherent"):
        for a, b in combinations([r for r in rows if r["prefix"] == prefix], 2):
            actual_sign = np.sign(a["real_goal"]-b["real_goal"])
            for kind in counts:
                predicted_sign = np.sign(a[kind+"_goal"]-b[kind+"_goal"])
                label = "tie" if not actual_sign or not predicted_sign else (
                    "agree" if actual_sign == predicted_sign else "reverse")
                counts[kind][label] += 1
    return {"schema": run.NAME+".analysis.v1", "rows": rows, "ranking": counts,
            "mean_open_mse": float(np.mean([x["open_mse"] for x in rows])),
            "mean_refresh_mse": float(np.mean([x["refresh_mse"] for x in rows])),
            "refresh_mse_lower_count": sum(x["refresh_mse"] < x["open_mse"] for x in rows),
            "scope": "one development root, dependent candidates, exact signs; no neural recomputation",
            "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(run.base.write_new(parser.parse_args().output, analyze()))
