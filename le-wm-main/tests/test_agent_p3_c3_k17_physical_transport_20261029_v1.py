import json
import pytest
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as t


def test_actual_disk_hashchain_and_tampering(tmp_path):
    path = tmp_path / 'journal.jsonl'
    j = t.Journal(path, 4096)
    first = j.emit('guard', {'accepted': True})
    second = j.emit('intent', {'step': 1})
    j.close()
    assert t.read_journal(path) == [first, second]
    path.write_bytes(path.read_bytes().replace(b'"step":1', b'"step":2'))
    with pytest.raises(ValueError, match='chain'):
        t.read_journal(path)


def test_exclusive_and_bounded_fail_closed(tmp_path):
    path = tmp_path / 'journal.jsonl'
    j = t.Journal(path, 1)
    with pytest.raises(ValueError, match='limit'):
        j.emit('intent', {})
    with pytest.raises(ValueError, match='fault'):
        j.emit('intent', {})
    j.close()
    assert path.read_bytes() == b''
    with pytest.raises(FileExistsError):
        t.Journal(path, 100)


@pytest.mark.parametrize('data', [b'{}', b'{}\n\n', b'{ "index": 0 }\n'])
def test_reject_incomplete_and_noncanonical(tmp_path, data):
    path = tmp_path / 'journal.jsonl'
    path.write_bytes(data)
    with pytest.raises((ValueError, KeyError, json.JSONDecodeError)):
        t.read_journal(path)


def test_census_not_sufficiency_proof():
    c = t.census()
    assert c['png_count'] == 6 * 3 * (11 + 6)
    assert c['png_bound_total'] + c['metadata_remaining'] + c['reserve'] == c['cap']
    assert c['metadata_sufficiency_proved'] is False
