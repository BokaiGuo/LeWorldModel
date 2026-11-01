import hashlib
import json

import pytest

from experiments.spectral_reachability.agent_p1_fixed_prefix_counts_20261029_v1 import load_verified, recount


def panel():
    candidate = {"terminal_Q": 0.5, "uphill_from_initial": True,
                 "better_terminal_than_original": True, "witness": True}
    return ({"rows": [{"result": {"initial_Q": 1.0, "stage_Q": 2.0,
                                  "baseline_Q": 1.0, "iterates": [candidate]}}]},
            {"group_denominator": 1, "candidate_denominator": 1})


def test_recounts_same_candidate_conjunction():
    report, receipt = panel()
    assert recount(report, receipt)["witness"] == 1


def test_rejects_wrong_saved_predicate():
    report, receipt = panel()
    report["rows"][0]["result"]["iterates"][0]["witness"] = False
    with pytest.raises(ValueError, match="predicates"):
        recount(report, receipt)


def test_rejects_wrong_denominator():
    report, receipt = panel()
    receipt["candidate_denominator"] = 2
    with pytest.raises(ValueError, match="denominator"):
        recount(report, receipt)


def test_rejects_tampered_report(tmp_path):
    report, receipt = panel()
    report_path, receipt_path = tmp_path / "report.json", tmp_path / "receipt.json"
    report_path.write_text(json.dumps(report))
    receipt.update(status="verified_model_only", complete=True,
                   report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest())
    receipt_path.write_text(json.dumps(receipt))
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    load_verified(report_path, receipt_path, receipt_sha)
    report_path.write_text("{}")
    with pytest.raises(ValueError, match="report SHA"):
        load_verified(report_path, receipt_path, receipt_sha)
