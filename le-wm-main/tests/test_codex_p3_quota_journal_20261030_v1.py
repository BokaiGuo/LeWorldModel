import pytest

from experiments.spectral_reachability import codex_p3_quota_journal_20261030_v1 as j


def test_durable_chain_and_exclusive(tmp_path):
    path = tmp_path / "journal"
    journal = j.Journal(path, 4096, 1024)
    journal.event({"kind": "attempt"})
    tail = journal.append({"kind": "failure"}, terminal=True)
    journal.close()
    assert [e["kind"] for e in j.read(path, tail, 4096)] == ["attempt", "failure"]
    with pytest.raises(FileExistsError):
        j.Journal(path, 4096, 1024)
    with pytest.raises(ValueError, match="external"):
        j.read(path, j.ZERO, 4096)


def test_tamper_and_truncation(tmp_path):
    path = tmp_path / "journal"
    writer = j.Journal(path, 4096, 1024)
    tail = writer.event({"kind": "attempt"})
    writer.close()
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"attempt", b"returnd"))
    with pytest.raises(ValueError, match="hash chain"):
        j.read(path, tail, 4096)
    path.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        j.read(path, tail, 4096)


def test_terminal_reserve(tmp_path):
    writer = j.Journal(tmp_path / "journal", 700, 400)
    with pytest.raises(ValueError, match="cap"):
        writer.event({"large": "x" * 400})
    writer.append({"kind": "failure"}, terminal=True)
    writer.close()
