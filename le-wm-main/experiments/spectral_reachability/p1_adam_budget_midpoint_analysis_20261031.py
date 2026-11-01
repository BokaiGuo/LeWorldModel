"""Audit midpoint latents and summarize same-prefix budget ranking descriptively."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_adam_budget_midpoint_cuda_20261031 as experiment

ROOT = experiment.ROOT
SOURCE = ROOT / "results/p1_adam_budget_midpoint_cuda_20261031_RUN01.json"
SOURCE_SHA = "0e9f40a5d3530f7ff36ade71fe41a3371f034a694b64092a006b8c8f81fb934b"
NAME = "p1_adam_budget_midpoint_analysis_20261031"
need = experiment.need


def analyze(data):
    rows = data["rows"]
    need(data["status"] == "complete" and len(rows) == 64 and len(data["stopped"]) == 8,
         "complete roster")
    lookup = {(row["index"], row["step"]): row for row in rows}
    need(len(lookup) == 64, "unique rows")
    for row in rows:
        pred = np.asarray(row["refresh_latent"], dtype=np.float64)
        actual = np.asarray(row["actual_latent"], dtype=np.float64)
        need(pred.shape == actual.shape == (192,) and np.isfinite(pred).all()
             and np.isfinite(actual).all()
             and abs(float(np.mean((pred-actual)**2))-row["refresh_error_mse"]) < 1e-6,
             "independent mse")
    comparisons = []
    summary = {}
    for low, high in ((5, 10), (10, 20), (5, 20)):
        group = []
        for i in range(24):
            if (i, low) not in lookup or (i, high) not in lookup:
                continue
            a, b = lookup[(i, low)], lookup[(i, high)]
            need(a["role"] == b["role"] and a["logical_pair"] == b["logical_pair"], "pair identity")
            delta = {key: b[key]-a[key] for key in ("open_goal_mse", "refresh_goal_mse", "actual_goal_mse")}
            item = {"index": i, "role": a["role"], "logical_pair": a["logical_pair"],
                    "steps": [low, high], "delta": delta}
            group.append(item)
            comparisons.append(item)
        summary[f"{low}-{high}"] = {
            "pairs": len(group),
            "open_reversed": sum(r["delta"]["open_goal_mse"]*r["delta"]["actual_goal_mse"] < 0 for r in group),
            "refresh_reversed": sum(r["delta"]["refresh_goal_mse"]*r["delta"]["actual_goal_mse"] < 0 for r in group),
        }
    return {"schema": NAME, "source_sha256": SOURCE_SHA, "pairs": comparisons, "summary": summary,
            "error_summary": {"count": len(rows),
                              "open_mean": float(np.mean([r["open_error_mse"] for r in rows])),
                              "refresh_mean": float(np.mean([r["refresh_error_mse"] for r in rows])),
                              "refresh_lower": sum(r["refresh_error_mse"] < r["open_error_mse"] for r in rows)},
            "P1_admission": False,
            "limitations": "Descriptive known-root repeated trajectories; horizon and observation differ."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    data = experiment.terminal.pair.pairing.read(SOURCE, SOURCE_SHA)
    result = analyze(data)
    result["analysis_sha256"] = experiment.terminal.pair.physical.sha(Path(__file__).resolve())
    need(experiment.terminal.pair.physical.sha(SOURCE) == SOURCE_SHA, "source drift")
    digest = experiment.terminal.pair.physical.prior.physical.design.run._write_new(out, result)
    print(json.dumps({"errors": result["error_summary"], "summary": result["summary"], "sha256": digest}))


if __name__ == "__main__":
    main()
