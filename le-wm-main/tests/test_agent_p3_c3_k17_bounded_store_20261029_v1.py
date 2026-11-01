import pytest
from experiments.spectral_reachability import agent_p3_c3_k17_bounded_store_20261029_v1 as m
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport


def test_integer_budget():
    c = m.census()
    assert c['png_count'] == 306
    assert c['ordinary_max'] + c['failure_reserve'] == c['total_max'] <= c['cap']
    assert 'not a guarantee' in c['scope']


def test_real_disk_full_chain_and_failure(tmp_path):
    s = m.Store(tmp_path / 'run')
    s.put('execution_binding.json', {'six': 6})
    a = s.emit('guard', {'accepted': True})
    b = s.emit('intent', {'step': 1})
    s.png('frame.png', b'fixture')
    s.put('report.json', {'status': 'fixture'})
    s.put('completion.json', {'status': 'fixture'})
    s.fail({'error': 'fixture'})
    assert transport.read_journal(s.path / 'journal.jsonl') == [a, b]
    assert (s.path / 'failure.json').is_file()
    with pytest.raises(ValueError, match='quota'):
        s.fail({'error': 'second'})


def test_quota_rejects_before_write_and_preserves_reserve(tmp_path, monkeypatch):
    monkeypatch.setitem(m.LIMITS, 'journal.jsonl', 1)
    s = m.Store(tmp_path / 'run')
    with pytest.raises(ValueError, match='quota'):
        s.emit('intent', {})
    assert not (s.path / 'journal.jsonl').exists()
    with pytest.raises(ValueError, match='store fault'):
        s.put('report.json', {})
    s.fail({'status': 'quota_failure'})
    assert (s.path / 'failure.json').exists()


def test_png_bound_and_exclusive(tmp_path):
    s = m.Store(tmp_path / 'run')
    with pytest.raises(ValueError, match='PNG cap'):
        s.png('a.png', b'x' * (m.PNG_MAX + 1))
    s.png('a.png', b'a')
    with pytest.raises(ValueError, match='quota'):
        s.png('a.png', b'b')
    with pytest.raises(ValueError, match='exclusive'):
        m.Store(tmp_path / 'run')


def test_symlink_and_unexpected_file_refused(tmp_path):
    s = m.Store(tmp_path / 'run')
    (s.path / 'evil').symlink_to(tmp_path)
    with pytest.raises(ValueError, match='unexpected'):
        s.emit('guard', {})
