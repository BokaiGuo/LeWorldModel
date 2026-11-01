"""Compare matched real and imagined midpoint plans on the fixed 24-row panel."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_p1_same_first5_feedback_compare_20261030_main as pair

NAME = "codex_p1_predicted_vs_real_10x2_20261030_main"
ROOT = pair.ROOT
PRED_DIR = ROOT/"results/codex_p1_two_stage_launcher_20261030_v1_run_pred10_MAIN01"
PRED_REPORT = "f48c90002a760d37ba0852383d0b533dfd4b8d89c87b1504141a75fad13bf0c1"
PRED_SAVED = ROOT/"results/codex_p1_predicted_midpoint_10x2_20261030_main_saved_MAIN01.json"
PRED_SHA = "d8f74f0c71aa8130ab774574a7de2befffd313c45d3c0d1895f382245e5f3bc3"


def compare(pred, real, pred_report, real_report, load, sidecar):
    pair.need(len(pred["rows"]) == len(real["rows"]) == len(pred_report["shards"])
              == len(real_report["shards"]) == 24, "full roster")
    rows = []
    for i, (p, r) in enumerate(zip(pred["rows"], real["rows"], strict=True)):
        pair.need(all(p[k] == r[k] for k in pair.IDS) and p["index"] == i, "identity")
        ps, rs = pred_report["shards"][i], real_report["shards"][i]
        pair.need(ps["index"] == rs["index"] == i, "shard index")
        pt, rt = load(i, ps, rs)
        pair.need(len(pt) >= 5 and len(rt) >= 5, "first five missing")
        for j in range(5):
            pair.need(pt[j]["step"] == rt[j]["step"] == j+1 and all(
                pt[j][k] == rt[j][k] for k in
                ("actual_bits", "state", "snapshot", "terminated", "truncated")), "first five mismatch")
        extra = sidecar(i, p)
        actual, imagined = [np.asarray(extra[k], dtype="<f4") for k in ("actual_latent", "predicted_latent")]
        pair.need(actual.shape == imagined.shape == (192,) and np.isfinite(actual).all()
                  and np.isfinite(imagined).all()
                  and hashlib.sha256(actual.tobytes()).hexdigest() == extra["actual_sha256"]
                  and hashlib.sha256(imagined.tobytes()).hexdigest() == extra["predicted_sha256"], "latent hash")
        pm, rm = p["endpoint_metrics"], r["endpoint_metrics"]
        delta = None if pm is None or rm is None else {k: rm[k]-pm[k] for k in pair.FIELDS}
        rows.append({"identity": {k: p[k] for k in pair.IDS}, "first_five_verified": True,
                     "predicted_status": p["status"], "real_status": r["status"],
                     "predicted_steps": p["accepted_suffix_steps"], "real_steps": r["accepted_suffix_steps"],
                     "predicted_endpoint": pm, "real_endpoint": rm, "real_minus_predicted": delta,
                     "midpoint_prediction_mse": float(np.mean((actual.astype(np.float64)-imagined)**2))})
    complete = [r for r in rows if r["real_minus_predicted"] is not None]
    return {"schema": NAME, "status": "descriptive_known_root_comparison", "rows": rows,
            "summary": {"rows": 24, "common_first5_verified": 24, "comparable": len(complete),
                        "real_lower_error": {k: sum(r["real_minus_predicted"][k] < 0 for r in complete)
                                             for k in pair.FIELDS},
                        "real_success": sum(bool(r["real_endpoint"]["native_success"]) for r in complete),
                        "predicted_success": sum(bool(r["predicted_endpoint"]["native_success"]) for r in complete),
                        "midpoint_prediction_mse_mean": sum(r["midpoint_prediction_mse"] for r in rows)/24},
            "independent_confirmation": False, "equal_total_budget_certified": False, "P1_admission": False,
            "limitations": ["known roots, repeated trajectories, post-outcome design",
                            "both encode real RGB but imagined arm discards encoding",
                            "imagined arm incurs one extra action encoder and prediction per second stage",
                            "prediction sidecars are bound but prediction not independently numerically replayed",
                            "not a complete resource or source provenance certification"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    out = Path(ap.parse_args().output)
    pair.need(out.is_absolute() and out.parent == ROOT/"results" and out.name.startswith(NAME+"_")
              and not out.exists() and not out.is_symlink(), "exclusive output")
    pred, real = pair.read(PRED_SAVED, PRED_SHA), pair.read(pair.FB_SAVED, pair.FB_SHA)
    pr = pair.read(PRED_DIR/"run_report.json", PRED_REPORT)
    rr = pair.read(pair.replay.FB_DIR/"run_report.json", pair.replay.FB_REPORT_SHA)
    def load(i, ps, rs):
        return (pair.transitions(PRED_DIR/f"row_{i:02d}"/"physical", ps["report_sha256"]),
                pair.transitions(pair.replay.FB_DIR/f"row_{i:02d}"/"physical", rs["report_sha256"]))
    def sidecar(i, row):
        return pair.read(PRED_DIR/f"row_{i:02d}"/"stage_2/predicted_midpoint.json",
                         row["midpoint_sidecar_sha256"])
    result = compare(pred, real, pr, rr, load, sidecar)
    result["sources"] = {str(PRED_SAVED): PRED_SHA, str(pair.FB_SAVED): pair.FB_SHA,
                         str(PRED_DIR/"run_report.json"): PRED_REPORT,
                         str(pair.replay.FB_DIR/"run_report.json"): pair.replay.FB_REPORT_SHA}
    result["code_sha256"] = pair.replay.physical.old.sha(Path(__file__).resolve())
    print(json.dumps({"sha256": pair.replay.physical.old.h10._write_new(out, result),
                      "summary": result["summary"]}))


if __name__ == "__main__":
    main()
