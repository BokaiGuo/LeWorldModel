import pytest

from experiments.spectral_reachability import codex_p1_bound_physical_reset_20261101_exclusive as m


def test_saved_journal():
    result = m.build()
    assert result["status"] == "verified_source_record"
    assert len(result["records"]) == 8
    assert all(r["completion_line"] and r["physical_root_equivalence"] == "unknown" for r in result["records"])
    assert [sum(r["source_slot"] == i for r in result["records"]) for i in (0, 1)] == [4, 4]
    assert not result["independent_roots_certified"]


def test_tampering_and_missing(tmp_path):
    root = m.original.ROOT / "results"
    report = (root / m.NAME / "report.json").read_bytes()
    journal = (root / m.NAME / "journal.jsonl").read_bytes()
    source = (root / m.SOURCE / "sourcepack.json").read_bytes()
    for args in ((report + b" ", journal, source), (report, journal[:-1], source),
                 (report, journal, source + b" ")):
        with pytest.raises(ValueError):
            m.verify(*args)
    assert m.build(tmp_path)["status"] == "unknown"


def test_exclusive_write(tmp_path, monkeypatch):
    import sys
    output = tmp_path / "output.json"
    output.write_text("untouched")
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(output)])
    with pytest.raises(FileExistsError):
        m.main()
    assert output.read_text() == "untouched"
