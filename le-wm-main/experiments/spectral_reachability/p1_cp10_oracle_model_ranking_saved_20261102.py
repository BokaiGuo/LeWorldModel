"""Pair fixed-bank model scores with sealed simulator results for diagnosis."""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from . import p1_cp10_oracle_model_ranking_20261102 as model

ROOT = model.ROOT
REPORT = ROOT / "results/p1_cp10_oracle_model_ranking_20261102_RUN_DEV01/report.json"
REPORT_SHA = "df3dfc9e51c35d59dc55eceea108257754c3e234e35d655de9058802254970eb"
CONTRACT = ROOT / "results/p1_cp10_oracle_model_ranking_20261102_PREP_DEV01.json"
CONTRACT_SHA = "37cd22e652cc3687250502093dbecc59a6a18ac6cab00974c43c557fb01dbb75"
need = model.require


def analyze():
    prev = model.saved.oracle.saved.physical.previous
    model_spec = prev.read(CONTRACT, CONTRACT_SHA)
    need(model_spec == model.protocol(), "model contract closure")
    report = prev.read(REPORT, REPORT_SHA)
    physical = prev.read(model.saved.RUN / "report.json", model.saved.REPORT_SHA)
    need(report["protocol_sha256"] == CONTRACT_SHA
         and report["model_before"] == report["model_after"] == model_spec["model_hash"]
         and [r["arm"] for r in report["rows"]] == [r["arm"] for r in physical["rows"]]
         == [r["arm"] for r in model_spec["rows"]], "model and physical roster")
    rows = []
    for mr, pr, spec in zip(report["rows"], physical["rows"], model_spec["rows"], strict=True):
        need(prev.read(REPORT.parent / (mr["arm"] + ".json")) == mr
             and mr["candidate_sha256"] == pr["candidate_sha256"] == spec["candidate_sha256"],
             "candidate and shard binding")
        model_score = np.asarray(mr["predicted_latent_costs"], dtype=np.float64)
        physical_score = np.asarray([r["terminal"]["score"] for r in pr["rows"]], dtype=np.float64)
        need(model_score.shape == physical_score.shape == (64,)
             and np.isfinite(model_score).all() and np.isfinite(physical_score).all(), "finite bank scores")
        mi, pi = int(np.argmin(model_score)), int(np.argmin(physical_score))
        need(mi == mr["model_argmin_index"] and pi == pr["oracle_selected_index"], "winner binding")
        rho = float(spearmanr(model_score, physical_score).statistic)
        need(np.isfinite(rho), "finite rank correlation")
        rows.append({"arm": mr["arm"], "oracle_index": pi,
                     "oracle_rank_by_model_competition": 1 + int(np.sum(model_score < model_score[pi])),
                     "model_index": mi,
                     "model_rank_by_oracle_competition": 1 + int(np.sum(physical_score < physical_score[mi])),
                     "model_selected_physical_score": float(physical_score[mi]),
                     "oracle_best_physical_score": float(physical_score[pi]),
                     "spearman_rho_descriptive": rho,
                     "candidate_count": len(model_score)})
    return {"schema": "p1_cp10_oracle_model_ranking_saved.v1", "rows": rows,
            "classification": "post_outcome_known_root_diagnostic",
            "P1_admission": False, "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(model.saved.oracle.saved.physical.previous.write_new(args.output, analyze()))
