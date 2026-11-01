import pytest

from experiments.spectral_reachability import codex_p1_root_replay_join_20261101_exclusive as m


def test_real_join():
    result = m.build()
    assert not result["unknown_sources"]
    assert len(result["diagnostic_replay"]["events"]) == 16
    assert all(r["completion_line"] and r["original_start"] for r in result["diagnostic_replay"]["events"])
    assert result["physical_root_deduplication"] == "unknown"
    assert not result["independent_roots_certified"]


def test_corruption_missing_and_inconsistent_baseline():
    path = m.original.ROOT / "results" / m.NAME
    report = (path / "report.json").read_bytes()
    journal = (path / "journal.jsonl").read_bytes()
    baseline = m.prior.build()["baseline"]["events"]
    with pytest.raises(ValueError):
        m.join(report, journal[:-1], baseline, "report", "journal")
    with pytest.raises(ValueError):
        m.join(report + b" ", journal, baseline, "report", "journal")
    with pytest.raises(ValueError):
        m.join(report, journal, [], "report", "journal")


def test_missing_stays_unknown(tmp_path):
    result = m.build(tmp_path)
    assert result["diagnostic_replay"] is None
    assert result["unknown_sources"]
    assert not result["P1_admitted"]


def test_exclusive_write(tmp_path, monkeypatch):
    import sys
    output = tmp_path / "result.json"
    output.write_text("keep")
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(output)])
    with pytest.raises(FileExistsError):
        m.main()
    assert output.read_text() == "keep"
