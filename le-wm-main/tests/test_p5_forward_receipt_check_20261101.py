import pytest

from experiments.spectral_reachability import p5_forward_receipt_check_20261101 as m


def test_receipt_recomputed_and_tampering_rejected(tmp_path, monkeypatch):
    s = m.s
    monkeypatch.setattr(s, "RESULTS", tmp_path)
    supervision = tmp_path / "supervision"
    supervision.mkdir()
    monkeypatch.setattr(s, "paths", lambda suffix: (supervision, tmp_path / "run"))
    monkeypatch.setattr(s, "preflight", lambda: ({}, {}))
    monkeypatch.setattr(s.hostgate, "load_report", lambda name, sha: {})
    review = {"result": "mock"}
    monkeypatch.setattr(s, "validate_supervisor", lambda *args: review)
    monkeypatch.setattr(s, "checked_review", lambda *args: review)
    preflight_sha = "a" * 64
    journal_sha = "b" * 64
    events = [{"preflight_sha256": preflight_sha, "preflight_name": "mock"}]
    raw = b"".join(s.bridge.canonical(e) for e in events)
    (supervision / "supervisor.jsonl").write_bytes(raw)
    supervisor_sha = s.bridge.digest(raw)
    acceptance = {"schema": s.NAME + ".accept", "status": "saved_self_report_consistency_not_scientific",
                  "producer_sha256": s.source_sha(), "probe_sha256": s.PROBE_SHA,
                  "contract_sha256": s.PREP_SHA, "preflight_sha256": preflight_sha,
                  "supervisor_sha256": supervisor_sha, "review": review, "P5_admission": False,
                  "boundary": "self-reported execution; finite differences not theoretical causality or fresh confirmation"}
    path = tmp_path / (s.NAME + "_accept_NEW.json")
    data = s.bridge.canonical(acceptance)
    path.write_bytes(data)
    digest = s.bridge.digest(data)
    assert m.check("NEW", preflight_sha, supervisor_sha, journal_sha, digest)["P5_admission"] is False
    acceptance["P5_admission"] = True
    data = s.bridge.canonical(acceptance)
    path.write_bytes(data)
    with pytest.raises(ValueError, match="acceptance content mismatch"):
        m.check("NEW", preflight_sha, supervisor_sha, journal_sha, s.bridge.digest(data))
    with pytest.raises(ValueError):
        m.check("NEW", preflight_sha, supervisor_sha, journal_sha, digest)


def test_bad_sha_rejected():
    with pytest.raises(ValueError, match="external SHA"):
        m.check("NEW", "bad", "b" * 64, "c" * 64, "d" * 64)
