"""Saved-artifact verification of fixed diag32 RGB diagnostic, no inference."""
import json
from pathlib import Path

import numpy as np

from . import p1_diag32_same_action_rgb_20261101_leibniz as diag

REPORT = diag.ROOT / "results/p1_diag32_same_action_rgb_20261101_leibniz_RUN_LEIBNIZ01/report.json"
REPORT_SHA = "5b52f5451a5f34a383987108113a72691514f3c99df7f5c8215aa2e6aa2d36c4"
PREP_SHA = "8e742482cf15cd31ed25cdd84e36acd5b6ee6c7029ac86f7a9ebd11c5806693a"


def verify():
    require, sha = diag.family.require, diag.family.sha
    require(sha(REPORT) == REPORT_SHA, "report hash")
    report = json.loads(REPORT.read_bytes())
    require(report["status"] == "complete" and report["failure"] is None and report["source_verified"]
            and report["model_before"] == report["model_after"] and not report["P1_admission"]
            and not report["training"] and not report["physical_execution"], "report status")
    prep = Path(report["prepared"])
    require(sha(prep) == PREP_SHA == report["prepare_sha256"] == report["binding_sha256"]
            and sha(REPORT.parent / "binding.json") == PREP_SHA, "prepare hash")
    value = json.loads(prep.read_bytes())
    require(value == json.loads((REPORT.parent / "binding.json").read_bytes()) == diag.build(), "source binding")
    require(report["model_before"] == value["model_hash"], "model binding")
    contract = diag.ledger_contract(value)
    account = diag.own.consumer.consume(report["ledger"]["journal"], report["ledger"]["sha256"],
                                        diag.own.spec.digest(contract), "structured", "node", diag.NAME)
    require(account == report["account"] and account["status"] == "closed_unverified"
            and not account["coverage_verified"] and account["charged_upper_bounds"] == diag.costs(), "account")
    result = report["result"]
    require(result["goal_latent"] == value["task"]["goal"], "goal")
    rows = diag.metrics(result["actual_latents"], result["predicted_latents"], result["goal_latent"],
                        value["physical_metrics"])
    require(rows == result["rows"], "metric recomputation")
    root = np.asarray(value["task"]["root_context"]["latent_history"][-1], np.float32)
    actual = np.asarray(result["actual_latents"][0], np.float32)
    predicted = np.asarray(result["predicted_latents"][0], np.float32)
    require(np.array_equal(root, predicted)
            and float(np.max(np.abs(root-actual))) == result["root_latent_max_abs_error"], "root")
    for row, key in zip((rows[0], rows[1], rows[3]), ("selected_q0", "selected_q1", "selected_q3"), strict=True):
        require(np.isclose(row["predicted_goal_mse"], value[key], rtol=1e-5, atol=1e-6), "saved score")
    return {"status": "saved_artifacts_verified_no_new_inference", "report_sha256": REPORT_SHA,
            "prepare_sha256": PREP_SHA, "ledger_sha256": report["ledger"]["sha256"],
            "steps": [row["step"] for row in rows], "candidate": 4, "iteration": 13,
            "full_cost_certified": False, "P1_admission": False}


if __name__ == "__main__":
    output = diag.ROOT / "results/p1_diag32_same_action_rgb_verify_20261101_leibniz_RUN01.json"
    print(json.dumps({"output": str(output), "sha256": diag.own.write(output, verify())}, sort_keys=True))
