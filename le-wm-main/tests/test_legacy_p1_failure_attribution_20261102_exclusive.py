import json
from pathlib import Path

import numpy as np

from experiments.empirical_spectral_audit import legacy_p1_failure_attribution_20261102_exclusive as a


def test_missing_external_source_is_visible():
    assert a.check_source({"provenance": {}})["state"] == "no_external_evidence_path"
    row = a.check_source({"provenance": {"external_evidence": "/not/a/real/evidence.json"}})
    assert row["state"] == "external_evidence_missing"


def test_changed_source_value_detected(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps({"action_risk": {"action_inverse_risk": 1},
                                "planning_consequence": {"planning_risk": 2}}))
    record = {"provenance": {"external_evidence": str(path)},
              "diagnostics": {"action_inverse_risk": 1},
              "heldout": {"planning_risk": 3}}
    assert a.check_source(record)["state"] == "source_values_mismatch"


def test_empty_and_rejected_inputs_not_silently_dropped(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    result = a.panel([bad], {"strata": []}, "fixture", 20261013)
    assert len(result["input"]) == len(result["load_failures"]) == 1
    assert result["recomputed"] is None


def test_historical_real_rows_reproduce():
    report = json.loads((a.OUTPUT / "report.json").read_text())
    assert [len(p["input"]) for p in report["panels"]] == [100, 140]
    for p in report["panels"]:
        assert p["input_stable"] and p["historical_key_set_matches"]
        assert not p["load_failures"]
        assert all(not row["different_fields"] for row in p["historical_stratum_comparisons"])
    assert [p["recomputed_summary"]["identifiable_strata"] for p in report["panels"]] == [7, 3]
    assert all(not p["recomputed_summary"]["claim_ready_for_positive_delta"] for p in report["panels"])


def test_rank_tie_behavior_is_source_order_not_average():
    # Documents inherited benchmark behavior; not an alternative selector.
    assert np.array_equal(a.benchmark._rank(np.array([1., 1., 2.])), [0., 1., 2.])
    assert a.clean({"v": float("nan")}) == {"v": None}
