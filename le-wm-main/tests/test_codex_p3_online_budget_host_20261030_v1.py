import pytest
from experiments.spectral_reachability import codex_p3_online_budget_host_20261030_v1 as host


def test_exclusive_output_outside_results_rejected(tmp_path):
    with pytest.raises(ValueError, match="exclusive"):
        host.output_path(tmp_path / (host.NAME + "_RUN_01"), "_RUN_")


def test_verify_refuses_changed_prepare(tmp_path, monkeypatch):
    path = tmp_path / "prepare.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="external prepare sha"):
        host.verify_prepare(path, "incorrect")


def test_report_written_when_journal_creation_fails(tmp_path, monkeypatch):
    import fcntl
    lock = tmp_path / "lock"
    lock.touch()
    monkeypatch.setattr(host, "LOCK", lock)
    monkeypatch.setattr(host.journal, "Journal", lambda *args: (_ for _ in ()).throw(OSError("disk")))
    fd = lock.open("r+")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        (tmp_path / "prepare.json").write_text("{}")
        result = host.child({}, [], [], tmp_path, fd.fileno(), {})
    finally:
        fd.close()
    assert result["status"] == "failed" and result["journal_tail"] is None
    assert result["error"]["type"] == "OSError"
    assert (tmp_path / "report.json").exists()
