import json

import pytest

from experiments import a0_pointmaze_replay_audit_20261102_codex_exclusive as audit


def test_real_replay_rebuild_matches_exclusive_result():
    expected = json.loads((audit.ROOT / "results" /
                           (audit.NAME + "_RESULT01.json")).read_bytes())
    actual = audit.build()
    assert actual == expected
    summary = actual["summary"]
    assert summary["archive_row_count"] == 144
    assert summary["position_pass"] and not summary["state_pass"]
    assert summary["fresh_reset_pass"] and summary["reused_reset_pass"]
    assert summary["saved_endpoint_pass"]
    failed = [(r["root"], r["frame"], r["actions"]) for r in actual["archive_rows"]
              if r["state_error"] > audit.STATE_TOLERANCE]
    assert failed == [(1900, 25, 1), (1900, 25, 2)]
    assert actual["protocol"]["mujoco_ticks_per_archive_action"] == 5
    assert not actual["scientific_admission"]


def test_tampered_prior_digest_rejected(monkeypatch):
    monkeypatch.setattr(audit, "PRIOR_SHA", "0" * 64)
    with pytest.raises(ValueError, match="sealed source/result identity"):
        audit.build()
