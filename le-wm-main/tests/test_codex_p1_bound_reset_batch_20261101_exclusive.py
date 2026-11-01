import pytest

from experiments.spectral_reachability import codex_p1_bound_reset_batch_20261101_exclusive as m


def test_saved_batch():
    result = m.build()
    assert len(result["verified"]) == len(m.PINS) == 11
    assert not result["unknown_sources"]
    assert result["reset_records"] == 72
    assert all(row["completion_line"] is not None for item in result["verified"] for row in item["records"])
    assert all(row["physical_root_equivalence"] == "unknown" for item in result["verified"] for row in item["records"])
    assert not result["independent_roots_certified"]


def test_corruption_and_missing(tmp_path):
    name = "p1_constructive_plan_physical_20261027_v1_main01"
    (source, source_sha), report_sha = m.PINS[name]
    directory = m.core.ROOT / "results" / name
    report = (directory / "report.json").read_bytes()
    journal = (directory / "journal.jsonl").read_bytes()
    source_raw = (directory.parent / source / "sourcepack.json").read_bytes()
    for inputs in ((report + b" ", journal, source_raw), (report, journal[:-1], source_raw),
                   (report, journal, source_raw + b" ")):
        with pytest.raises(ValueError):
            m.verify(*inputs, report_sha, source_sha)
    result = m.build(tmp_path, {name: m.PINS[name]})
    assert not result["verified"] and result["unknown_sources"][0]["status"] == "unknown"


def test_exclusive_write(tmp_path, monkeypatch):
    import sys
    path = tmp_path / "result.json"
    path.write_text("preserved")
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(path)])
    with pytest.raises(FileExistsError):
        m.main()
    assert path.read_text() == "preserved"
