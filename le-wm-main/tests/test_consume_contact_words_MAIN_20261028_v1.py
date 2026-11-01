import json
import pytest
from experiments.spectral_reachability import consume_contact_words_MAIN_20261028_v1 as m


def journal(tmp_path, events):
    path = tmp_path / "journal.jsonl"
    previous = None
    lines = []
    for sequence, event in enumerate(events):
        row = dict(sequence=sequence, previous_sha256=previous, event=event)
        previous = m.producer.digest(row)
        lines.append(json.dumps(dict(row, entry_sha256=previous)).encode() + b"\n")
    path.write_bytes(b"".join(lines))
    return path


def test_full_large_journal_and_no_original_mutation(tmp_path):
    event = dict(payload="x" * (8 * 1024**2 + 1))
    path = journal(tmp_path, [event])
    assert m.read_journal(path) == [event]
    _, _, _, physical = m.producer.helpers()
    original = physical.read_journal
    proxy = m.ReaderBridge(physical)
    assert proxy.read_journal(path) == [event]
    assert physical.read_journal is original
    assert proxy.validate_frame is physical.validate_frame


def test_hash_chain_tampering_rejected(tmp_path):
    path = journal(tmp_path, [dict(value=1)])
    path.write_bytes(path.read_bytes().replace(b'"value": 1', b'"value": 2'))
    with pytest.raises(ValueError, match="journal chain"):
        m.read_journal(path)


def test_partial_and_cap_rejected(tmp_path, monkeypatch):
    path = journal(tmp_path, [dict(value=1)])
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    with pytest.raises(ValueError, match="complete journal line"):
        m.read_journal(path)
    monkeypatch.setattr(m, "JOURNAL_CAP", 1)
    with pytest.raises(ValueError, match="journal bound"):
        m.read_journal(path)
