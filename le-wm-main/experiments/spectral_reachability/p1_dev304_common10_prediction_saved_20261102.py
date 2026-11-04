"""Analyze sealed same-action prediction and physical outcomes."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from . import p1_dev304_common10_prediction_20261102 as source

base, saved = source.base, source.saved
CONTRACT_SHA = "282320bb65dff187d1da5a4d9f63e6a7d12daf10500ee5b13c9b764f28cd6665"
REPORT_SHA = "78be0ebe24043df0e325e18aba107c06c86dcc6b4b6ca6229615ed24d518b594"
ROOT = source.ROOT/"results"
RUN = ROOT/f"{source.NAME}_RUN_DEV01"
need = source.need


def mse(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    need(a.shape == b.shape == (192,) and np.isfinite(a).all() and np.isfinite(b).all(), "latent")
    return float(np.mean((a-b)**2))


def analyze():
    spec = base.read(ROOT/f"{source.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    need(spec == source.protocol(), "contract")
    report = base.read(RUN/"report.json", REPORT_SHA)
    need(report["contract_sha256"] == CONTRACT_SHA
         and report["model_before"] == report["model_after"] == spec["model_hash"], "model report")
    need(report["ledger"]["encoder"] == report["ledger"]["projector"] == {"calls": 31, "samples": 31}
         and report["ledger"]["action_encoder"] == report["ledger"]["predictor"] == {"calls": 18, "samples": 18},
         "ledger")
    physical = saved.analyze()
    need([r["arm"] for r in report["rows"]] == [x["arm"] for x in spec["branches"]], "roster")
    rows = []
    for row, branch, truth in zip(report["rows"], spec["branches"], physical["rows"], strict=True):
        need(base.read(RUN/(row["arm"]+".json")) == row and row["tail_sha256"] == branch["tail_sha256"]
             and row["prefix"] == truth["prefix"] and row["tail"] == truth["tail"], "row binding")
        v = row["vectors"]
        need(set(v) == {"goal", "real5", "real10", "pred5", "open10", "refresh10"}, "vector roster")
        rows.append({"arm": row["arm"], "prefix": row["prefix"], "tail": row["tail"],
                     "pred5_mse": mse(v["pred5"], v["real5"]),
                     "open10_mse": mse(v["open10"], v["real10"]),
                     "refresh10_mse": mse(v["refresh10"], v["real10"]),
                     "open_goal_cost": mse(v["open10"], v["goal"]),
                     "refresh_goal_cost": mse(v["refresh10"], v["goal"]),
                     "real_goal_cost": mse(v["real10"], v["goal"]),
                     "physical_position_error": truth["terminal"]["object_position_error"],
                     "physical_angle_error": truth["terminal"]["object_angle_error"]})
    pairs = []
    for prefix in ("original", "coherent"):
        subset = [r for r in rows if r["prefix"] == prefix]
        for a, b in combinations(subset, 2):
            def sign(key):
                return int(np.sign(a[key] - b[key]))
            pairs.append({"prefix": prefix, "tails": [a["tail"], b["tail"]],
                          "open": sign("open_goal_cost"), "refresh": sign("refresh_goal_cost"),
                          "real_latent": sign("real_goal_cost"), "physical_position": sign("physical_position_error"),
                          "physical_angle": sign("physical_angle_error")})
    return {"schema": source.NAME+".saved.v1", "rows": rows, "paired_rankings": pairs,
            "mean_open10_mse": float(np.mean([x["open10_mse"] for x in rows])),
            "mean_refresh10_mse": float(np.mean([x["refresh10_mse"] for x in rows])),
            "open_vs_real_agreement": sum(x["open"] == x["real_latent"] != 0 for x in pairs),
            "refresh_vs_real_agreement": sum(x["refresh"] == x["real_latent"] != 0 for x in pairs),
            "real_vs_position_agreement": sum(x["real_latent"] == x["physical_position"] != 0 for x in pairs),
            "classification": "post_outcome_one_known_root", "independent_confirmation": False,
            "p1_admission": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    print(base.write_new(p.parse_args().output, analyze()))
