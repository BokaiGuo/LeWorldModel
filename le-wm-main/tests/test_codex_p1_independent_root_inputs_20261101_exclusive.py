import pytest

from experiments.spectral_reachability import codex_p1_independent_root_inputs_20261101_exclusive as m


def test_saved_inputs():
    result = m.build()
    assert result["baseline"]["reset_records"] == 44
    assert len(result["additional_overlay"]["events"]) == 12
    assert [len(g["overlay_event_indices"]) for g in result["dedup_input_source_slot_groups"]] == [6, 6]
    assert not result["unknown_sources"]
    assert not result["independent_roots_certified"]
    assert result["physical_root_deduplication"] == "unknown"


def test_tampered_or_truncated_journal_rejected():
    path = m.original.ROOT / "results" / m.OVERLAY
    report = (path / "report.json").read_bytes()
    journal = (path / "journal.jsonl").read_bytes()
    prepare = (path.parent / "p1_moving_target_overlay_source_20261027_v1_prepare_main01.json").read_bytes()
    with pytest.raises(ValueError):
        m.overlay(report, journal[:-1], "report", "journal", prepare)
    with pytest.raises(ValueError):
        m.overlay(report, journal.replace(b"202798201", b"202798202", 1), "report", "journal", prepare)
    with pytest.raises(ValueError):
        m.overlay(report + b" ", journal, "report", "journal", prepare)
    with pytest.raises(ValueError):
        m.overlay(report, journal, "report", "journal", prepare + b" ")


def test_missing_inputs_remain_unknown(tmp_path):
    result = m.build(tmp_path)
    assert result["additional_overlay"] is None
    assert not result["dedup_input_source_slot_groups"]
    assert result["unknown_sources"]
    assert not result["P1_admitted"]


def test_exclusive_output(tmp_path, monkeypatch):
    import sys
    output = tmp_path / "out.json"
    output.write_text("keep")
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(output)])
    with pytest.raises(FileExistsError):
        m.main()
    assert output.read_text() == "keep"
