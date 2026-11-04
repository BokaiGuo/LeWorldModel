"""Independent vector/ledger verification of the post-outcome CUDA audit."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_second_feedback_prediction_audit_20261102 as audit

ROOT = audit.ROOT
RUN = ROOT / "results/p1_second_feedback_prediction_audit_20261102_RUN_DEV01"
CONTRACT = ROOT / "results/p1_second_feedback_prediction_audit_20261102_PREP_DEV01.json"
CONTRACT_SHA = "f203092b496620fe197e09ef59a7cde9894a83daa3ddcb66037f50c3d40fc7a2"
REPORT_SHA = "d7e9d477757998c70e6afe1b9d105ea359fec9181a1659cb8b00b5a03a769d92"
need = audit.require


def analyze():
    contract = audit.planner.first.fixed(CONTRACT, CONTRACT_SHA)
    need(contract == audit.protocol(), "contract and source closure")
    report = audit.planner.first.fixed(RUN / "report.json", REPORT_SHA)
    need(report["schema"] == audit.NAME + ".report.v1"
         and report["protocol_sha256"] == CONTRACT_SHA
         and report["model_before"] == report["model_after"] == contract["model_hash"]
         and [r["arm"] for r in report["rows"]] == [r["arm"] for r in contract["rows"]],
         "model and roster")
    goal = np.asarray(contract["task"]["goal"], dtype=np.float64)
    rows = []
    for row, expected in zip(report["rows"], contract["rows"], strict=True):
        need(audit.planner.first.fixed(RUN / (row["arm"] + ".json"),
                                      audit.planner.first.file_sha(RUN / (row["arm"] + ".json"))) == row,
             "row shard")
        for name in ("encoder", "projector"):
            need(report["hook_ledger"][row["arm"] + ":encode:" + name] == {"calls": 4, "samples": 4},
                 "encoding count")
        for name in ("action_encoder", "predictor"):
            need(report["hook_ledger"][row["arm"] + ":predict:" + name] == {"calls": 2, "samples": 2},
                 "predictor count")
        vectors = {k: np.asarray(v, dtype=np.float64) for k, v in row["vectors"].items()}
        need(set(vectors) == {"pred_old", "pred_new", "actual_old", "actual_new"}
             and all(v.shape == goal.shape and np.isfinite(v).all() for v in vectors.values()),
             "vector shape and finiteness")
        score = {k: float(np.mean((v-goal)**2)) for k, v in vectors.items()}
        need(all(np.isclose(score[k], row["goal_mse"][k], rtol=1e-6, atol=2e-7) for k in score),
             "goal-score recomputation")
        err = {kind: float(np.mean((vectors["pred_"+kind] - vectors["actual_"+kind])**2))
               for kind in ("old", "new")}
        need(all(np.isclose(err[k], row["prediction_mse"][k], rtol=1e-6, atol=2e-7) for k in err),
             "prediction MSE recomputation")
        ranks = audit.classify(*[score[k] for k in ("pred_old", "pred_new", "actual_old", "actual_new")])
        need(ranks["ranking"] == row["ranking"]["ranking"]
             and row["physical_evaluation_only"] == expected["physical_evaluation_only"], "ranking and physical binding")
        rows.append({"arm": row["arm"], "recomputed_goal_mse": score, "recomputed_prediction_mse": err,
                     "ranking": ranks, "physical": row["physical_evaluation_only"]})
    return {"schema": "p1_second_feedback_prediction_saved.v1", "rows": rows,
            "source_report_sha256": REPORT_SHA, "post_outcome_development_only": True,
            "P1_admission": False, "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(audit.planner.first.write_new(args.output, analyze()))
