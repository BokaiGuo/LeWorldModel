import pytest

from experiments.spectral_reachability import e3_cell02_readiness_handoff_20261102_codexexclusive as h


def test_failure_remains_visible_and_cannot_authorize(monkeypatch):
    def fail():
        raise ValueError("source drift")
    monkeypatch.setattr(h, "history_check", fail)
    monkeypatch.setattr(h.readiness, "cell02", lambda: {"saved_setup_joined": True})
    monkeypatch.setattr(h.gate, "audit", lambda: {"production_ready": False})
    result = h.build()
    assert result["checks"]["multitask_saved_history"]["status"] == "failed"
    assert result["checks"]["multitask_saved_history"]["error"] == "source drift"
    assert result["dispatch_ready"] is False
    assert result["scientific_admission"] is False


def test_output_exclusive(monkeypatch, tmp_path):
    monkeypatch.setattr(h, "ROOT", tmp_path)
    monkeypatch.setattr(h, "build", lambda: {"checks": {}})
    path = tmp_path / (h.NAME + "_test.json")
    path.write_text("existing")
    monkeypatch.setattr("sys.argv", ["handoff", "--output", str(path)])
    with pytest.raises(FileExistsError):
        h.main()
    assert path.read_text() == "existing"


def test_missing_or_changed_history_rejected(monkeypatch, tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{}")
    monkeypatch.setattr(h, "HISTORY", path)
    with pytest.raises(ValueError, match="hash mismatch"):
        h.history_check()
