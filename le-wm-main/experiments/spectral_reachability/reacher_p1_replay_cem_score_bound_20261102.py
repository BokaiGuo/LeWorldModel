"""Check replayed winner costs against the original CEM elite-mean upper bound."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import reacher_p1_52_q_metric_analysis_20261102 as qanalysis
from . import reacher_p1_52_replay_contract_20261102_codexexclusive as contract
from . import reacher_pc_p1_fd_cp5_host_20261102_codexexclusive as host
from . import reacher_pc_p1_fd_cp5_saved_20261102_codexexclusive as saved

TOLERANCE = 1e-5


def score_bounds(receipt, original, replay, matrices):
    qanalysis.pairwise.analyze(receipt)
    results = []
    for row in receipt["rows"]:
        key = (row["seed"], row["arm"])
        source = [x for x in original[row["group"]]
                  if x["kind"] == "arm_returned" and (x["seed"], x["arm"]) == key]
        inference = [x for x in replay if x["kind"] == "model_returned"
                     and x.get("phase") == "prediction"
                     and (x["seed"], x["arm"], x["group"]) == (*key, row["group"])]
        host.require(len(source) == len(inference) == 1, "source or replay roster")
        receipt0 = source[0]["result"]["suffix_receipt"]
        trace = receipt0["trace"]
        host.require(receipt0["final_selection"] == "best_sample" and len(trace) == 2
                     and np.isfinite(trace).all(), "CEM trace contract")
        p = np.asarray(inference[0]["predicted"], np.float64)
        target = np.asarray(inference[0]["goal"], np.float64)
        host.require(p.ndim == 1 and p.shape == target.shape and p.size > 0
                     and np.isfinite(p).all() and np.isfinite(target).all(), "prediction vectors")
        diff = p - target
        if row["group"] == "Q":
            q = matrices[row["seed"]]
            host.require(q.shape == (p.size, p.size), "Q dimensions")
            diff = q @ diff
        score = float(np.dot(diff, diff) / p.size)
        bound = float(trace[-1])
        results.append({"group": row["group"], "seed": row["seed"], "arm": row["arm"],
                        "replayed_score": score, "elite_mean_bound": bound,
                        "excess": score - bound, "within_bound": score <= bound + TOLERANCE})
    return {"rows": results, "count": len(results),
            "violations": sum(not x["within_bound"] for x in results),
            "tolerance_absolute": TOLERANCE,
            "scope": ("necessary upper-bound check only; elite mean is not the original selected score; "
                      "no proof of exact numerical identity, independence or P1"),
            "P1_admission": False}


def main():
    parser = argparse.ArgumentParser()
    for name in ("receipt", "receipt-sha", "replay", "replay-sha", "l2", "l2-sha",
                 "q", "q-sha", "q-plan", "q-plan-sha", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    inputs = ((args.receipt, args.receipt_sha), (args.replay, args.replay_sha),
              (args.l2, args.l2_sha), (args.q, args.q_sha), (args.q_plan, args.q_plan_sha))
    for path, sha in inputs:
        host.require(contract.sha(path) == sha, "input hash mismatch")
    receipt = json.loads(Path(args.receipt).read_text())
    host.require(receipt["journal_sha256"] == args.replay_sha, "replay binding")
    plan = json.loads(Path(args.q_plan).read_text())
    qpath = plan["q_report"]
    host.require(contract.sha(qpath) == plan["files"][qpath], "Q source hash")
    qrows = json.loads(Path(qpath).read_text())["rows"]
    host.require([x["seed"] for x in qrows] == [20263160, 20263161], "Q roster")
    matrices = {x["seed"]: qanalysis.metric_sqrt(x["goal_metric"]["gram"]) for x in qrows}
    result = score_bounds(receipt, {"L2": saved.verify_journal(args.l2), "Q": saved.verify_journal(args.q)},
                          saved.verify_journal(args.replay), matrices)
    result["input_sha256"] = {path: sha for path, sha in inputs}
    host.write_new(args.output, result)
    print(json.dumps({"count": result["count"], "violations": result["violations"],
                      "sha256": contract.sha(args.output)}))


if __name__ == "__main__":
    main()
