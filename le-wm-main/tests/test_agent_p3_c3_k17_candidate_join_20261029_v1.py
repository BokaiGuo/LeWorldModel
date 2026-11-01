import hashlib
import pytest
from experiments.spectral_reachability import agent_p3_c3_k17_candidate_join_20261029_v1 as m


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_keys_are_type_distinguished():
    assert m.normalize({1: 'a'}) != m.normalize({'1': 'a'})
    assert m.normalize({2: 'b', 1: 'a'}) == m.normalize({1: 'a', 2: 'b'})


def test_fixture_join_is_saved_projection_not_physical_authority(tmp_path, monkeypatch):
    monkeypatch.setattr(m.outer, 'material', lambda: ({'roots': [1, 2]}, [({'slot': s, 'root': r, 'mode': mode, 'K': 17}, {}, {}) for s, r, mode in m.outer.ORDER]))
    def execute(data, sources, store, factory):
        rows = m.outer.make_rows()
        for row in rows:
            row.update(status='horizon_completed', result={'status': 'horizon_completed', 'returned': 5})
            store.emit('branch_closed', row=row.copy())
        return {'rows': rows, 'fixed_denominator': 6, 'errors': [], 'production_authorized': False, 'physical_release': False}
    monkeypatch.setattr(m.outer, 'execute_material', execute)
    path = tmp_path/'candidate'
    receipt = m.run_fixture(path, lambda: None)
    result = m.consume(path, receipt['manifest_sha256'], verify_sources=True)
    assert result['lifecycle_verified'] and result['source_snapshot_matches_current']
    assert not result['native_provenance_verified'] and not result['physical_release']
    assert [r['status'] for r in result['rows']] == ['horizon_completed'] * 6
    with pytest.raises(ValueError):
        m.run_fixture(path, lambda: None)
    (path/'outer'/'journal.jsonl').write_bytes(b'tampered\n')
    with pytest.raises(ValueError):
        m.consume(path, receipt['manifest_sha256'])


def test_sources_rebuilt_not_trusted_from_saved_snapshot(tmp_path, monkeypatch):
    def material():
        return {'roots': [1, 2]}, [({'slot': s, 'root': r, 'mode': mode, 'K': 17}, {}, {}) for s, r, mode in m.outer.ORDER]
    monkeypatch.setattr(m.outer, 'material', material)
    def execute(data, sources, store, factory):
        rows = m.outer.make_rows()
        rows[0].update(status='engineering_failed', error={'type': 'ValueError', 'message': 'fixture'})
        store.emit('branch_closed', row=rows[0].copy())
        return {'rows': rows, 'fixed_denominator': 6, 'errors': [1], 'production_authorized': False, 'physical_release': False}
    monkeypatch.setattr(m.outer, 'execute_material', execute)
    receipt = m.run_fixture(tmp_path/'candidate', lambda: None)
    monkeypatch.setattr(m.outer, 'material', lambda: ({'roots': [9, 2]}, material()[1]))
    with pytest.raises(ValueError, match='source snapshot'):
        m.consume(tmp_path/'candidate', receipt['manifest_sha256'], verify_sources=True)
    assert not m.consume(tmp_path/'candidate', receipt['manifest_sha256'])['source_snapshot_matches_current']
