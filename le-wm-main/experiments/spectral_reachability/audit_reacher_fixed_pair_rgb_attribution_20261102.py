"""Check saved attribution bindings and calculate root-level rank counts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import reacher_fixed_pair_rgb_attribution_20261102 as run


def sign(value):
    if not math.isfinite(value):
        raise ValueError("nonfinite score")
    return (value > 0) - (value < 0)


def summarize(plan, report, source_rows, replay_rows, digest):
    need = run.host.require
    need(report["schema"] == "reacher_fixed_pair_rgb_attribution_result_v1"
         and report["plan_sha256"] == digest and report["P1_admission"] is False
         and report["model_before"] == report["model_after"] == plan["model_hash"], "report binding")
    need(len(report["rows"]) == len(source_rows) == 26 and len(replay_rows) == 52, "denominator")
    counts = {}
    for index, (row, predicted) in enumerate(zip(report["rows"], source_rows, strict=True)):
        seed, prefix = row["seed"], row["prefix"]
        need((seed, prefix) == (predicted["seed"], predicted["prefix"]), "row identity")
        actual = row["actual"]
        saved = replay_rows[index*2:index*2+2]
        need(len(actual) == 2 and [x["label"] for x in actual] == ["l2", "direction"], "pair order")
        for a, s in zip(actual, saved, strict=True):
            need((s["seed"], s["prefix"], s["label"]) == (seed, prefix, a["label"])
                 and a["final_rgb_sha256"] == s["final_rgb_sha256"]
                 and a["physical_final"] == s["final_distance"], "physical binding")
        for key in ("l2", "direction"):
            need(row["model_"+key] == predicted[key+"_model_cost"], "prediction binding")
        group = counts.setdefault(str(seed), {"pairs": 0, "physical_ties": 0,
                                               "l2": [0, 0, 0, 0], "direction": [0, 0, 0, 0]})
        group["pairs"] += 1
        physical = sign(actual[1]["physical_final"] - actual[0]["physical_final"])
        group["physical_ties"] += physical == 0
        for key in ("l2", "direction"):
            model = sign(row["model_"+key][1] - row["model_"+key][0])
            real = sign(actual[1]["real_"+key] - actual[0]["real_"+key])
            values = group[key]
            values[0] += model * real < 0
            values[1] += real * physical < 0
            values[2] += model == 0
            values[3] += real == 0
    need(set(counts) == {"20263160", "20263161"}
         and all(x["pairs"] == 13 for x in counts.values()), "root roster")
    return {"schema": "reacher_fixed_pair_rgb_attribution_saved_v1", "status": "saved_consistency_complete",
            "columns": ["model_vs_real_reversals", "real_vs_physical_reversals", "model_ties", "real_ties"],
            "roots": counts, "P1_admission": False,
            "limits": ["model scores and latent encodings not independently recomputed",
                       "two exposed roots, paired arms dependent", "no planning advantage established"]}


def main():
    p = argparse.ArgumentParser()
    for key in ("plan", "plan-sha", "report", "report-sha", "output"):
        p.add_argument("--" + key, required=True)
    args = p.parse_args()
    plan_path, report_path = Path(args.plan), Path(args.report)
    need = run.host.require
    need(run.setup._sha256_file(plan_path) == args.plan_sha
         and run.setup._sha256_file(report_path) == args.report_sha, "input hash")
    plan = json.loads(plan_path.read_text())
    need(plan == run.prepare(), "source drift")
    result = summarize(plan, json.loads(report_path.read_text()), json.loads(run.PRED.read_text())["rows"],
                       json.loads(run.RECEIPT.read_text())["rows"], args.plan_sha)
    result.update(plan_sha256=args.plan_sha, report_sha256=args.report_sha,
                  verifier_sha256=run.setup._sha256_file(Path(__file__)))
    run.host.write_new(args.output, result)
    print(json.dumps(result["roots"], sort_keys=True))


if __name__ == "__main__":
    main()
