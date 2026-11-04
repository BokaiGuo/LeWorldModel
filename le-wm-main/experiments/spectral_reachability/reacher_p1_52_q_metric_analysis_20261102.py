"""Post-hoc fixed-action Q-metric diagnostic; no model or physics execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import reacher_p1_52_pairwise_analysis_20261102 as pairwise
from . import reacher_pc_p1_fd_cp5_host_20261102_codexexclusive as host
from . import reacher_pc_p1_fd_cp5_saved_20261102_codexexclusive as saved
from . import reacher_p1_52_replay_contract_20261102_codexexclusive as contract


def metric_sqrt(gram):
    gram = np.asarray(gram, dtype=np.float64)
    host.require(gram.ndim == 2 and gram.shape[0] == gram.shape[1] and gram.shape[0] > 0
                 and np.isfinite(gram).all()
                 and np.allclose(gram, gram.T, rtol=1e-12, atol=1e-12), "invalid Gram")
    values, vectors = np.linalg.eigh(gram)
    host.require(values.min() >= -1e-10 * max(1.0, np.abs(values).max()), "Gram not PSD")
    return (vectors * np.sqrt(0.01 / (np.maximum(values, 0) + 0.01))[None, :]) @ vectors.T


def analyze(receipt, events, matrices):
    pairwise.analyze(receipt)  # Require a complete fixed roster and finite existing metrics.
    rows = []
    for row in receipt["rows"]:
        key = (row["group"], row["seed"], row["arm"])
        scoped = [e for e in events if (e.get("group"), e.get("seed"), e.get("arm")) == key]
        pred = [e for e in scoped if e["kind"] == "model_returned" and e.get("phase") == "prediction"]
        actual = [e for e in scoped if e["kind"] == "model_returned" and e.get("phase") == "actual_rgb_encode"]
        host.require(len(pred) == len(actual) == 1 and row["seed"] in matrices, "latent event roster")
        p, a, g = (np.asarray(x, np.float64) for x in (pred[0]["predicted"], actual[0]["actual"], pred[0]["goal"]))
        q = matrices[row["seed"]]
        host.require(p.shape == a.shape == g.shape == (q.shape[0],)
                     and all(np.isfinite(x).all() for x in (p, a, g)), "latent shape or finiteness")
        values = {"predicted_goal_distance": float(np.linalg.norm(q @ (p - g))),
                  "actual_rgb_goal_distance": float(np.linalg.norm(q @ (a - g)))}
        rows.append({**row, **values})
    result = pairwise.analyze({**receipt, "rows": rows})
    for panel in result["panels"]:
        panel["mean_l2_prediction_error"] = panel.pop("mean_prediction_error")
    result["schema"] = "p1_52_q_metric_pairwise_analysis_v1"
    result["metric"] = "sqrt((z-goal)^T 0.01*(Gram+0.01 I)^-1 (z-goal))"
    result["scope"] = ("post-hoc on two exposed roots; Q built from frozen goal encoder; "
                       "not independent samples or proof of P1")
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("receipt", "receipt-sha", "journal", "journal-sha", "plan", "plan-sha", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    for path, sha in ((args.receipt, args.receipt_sha), (args.journal, args.journal_sha),
                      (args.plan, args.plan_sha)):
        host.require(contract.sha(path) == sha, "input SHA mismatch")
    receipt = json.loads(Path(args.receipt).read_text())
    host.require(receipt["journal_sha256"] == args.journal_sha, "journal binding")
    plan = json.loads(Path(args.plan).read_text())
    qpath = plan["q_report"]
    host.require(contract.sha(qpath) == plan["files"][qpath], "goal metric source hash")
    qreport = json.loads(Path(qpath).read_text())
    host.require([r["seed"] for r in qreport["rows"]] == [20263160, 20263161], "Q roots")
    matrices = {r["seed"]: metric_sqrt(r["goal_metric"]["gram"]) for r in qreport["rows"]}
    result = analyze(receipt, saved.verify_journal(args.journal), matrices)
    result.update(receipt_sha256=args.receipt_sha, journal_sha256=args.journal_sha,
                  q_plan_sha256=args.plan_sha, q_report_sha256=plan["files"][qpath])
    host.write_new(args.output, result)
    print(json.dumps({"output": args.output, "sha256": contract.sha(args.output)}))


if __name__ == "__main__":
    main()
