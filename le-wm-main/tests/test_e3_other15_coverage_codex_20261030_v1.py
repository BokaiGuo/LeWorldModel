import hashlib
import json

import pytest

from experiments.spectral_reachability import e3_other15_coverage_codex_20261030_v1 as m


def test_matrix_fixed_order_and_hash(tmp_path):
    path = tmp_path/'resolver.json'
    ids = [f'r{r}.g{g}.h{h}.d{d}' for r in range(2) for g in range(2)
           for h in (1, 3) for d in ('Dxy', 'Dx')]
    obj = {'denominator': {'cells': 16}, 'cells': [
        {'index': i, 'cell_id': x} for i, x in enumerate(ids)]}
    path.write_text(json.dumps(obj))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(m.matrix(path, sha)) == 16
    obj['cells'][2]['cell_id'] = obj['cells'][0]['cell_id']
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='hash mismatch'):
        m.matrix(path, sha)
    with pytest.raises(ValueError, match='identity mismatch'):
        m.matrix(path, m.digest(path))


def create_job(path, *, cell=1, terminal='completed20', nodes=(10, 15), status='completed'):
    store = m.transport.Store(path)
    store.event('physical', 'record', 'launcher_authority_verified', store.index, {'cell': cell})
    for node in nodes:
        store.event('physical', 'record', 'native_e3_node_return', store.index, {'boundary': node})
    if terminal:
        store.event('physical', 'record', 'launcher_guarded_return', store.index,
                    {'cell': cell, 'status': terminal})
    store.publish(status)


def test_transport_job_partial_and_early_stop(tmp_path):
    early = tmp_path/'early'
    create_job(early, terminal='native_stopped_prefix', nodes=())
    result = m.job(early)
    assert result['publication_complete'] is True
    assert result['terminal_status_recorded'] == 'native_stopped_prefix'
    assert result['feedback_node_boundaries_recorded'] == []
    assert not result['scientific_admission'] and not result['full_saved_strict']
    partial = tmp_path/'partial'
    create_job(partial, terminal=None, nodes=(), status='partial')
    assert m.job(partial)['publication_complete'] is False
    assert m.job(partial)['terminal_status_recorded'] is None


def test_wrong_boundary_order_rejected(tmp_path):
    path = tmp_path/'job'
    create_job(path, nodes=(15, 10))
    with pytest.raises(ValueError, match='boundary sequence'):
        m.job(path)


def test_excluded_cell_never_opened_duplicate_jobs_stay_one_row(tmp_path, monkeypatch):
    cells = [{'index': i, 'cell_id': str(i), 'status': 'prepared'} for i in range(16)]
    monkeypatch.setattr(m, 'matrix', lambda *_: cells)
    monkeypatch.setattr(m, 'authorities', lambda *_: {})
    monkeypatch.setattr(m, 'numerical', lambda _, i: {'state': str(i)})
    prefix = 'agent_e3_chunked_identity_host_20261029_v1_'
    for name in ('CELL03_MAIN01', 'MAIN05', 'MAIN06'):
        (tmp_path/(prefix+name)).mkdir()
    seen = []
    def fake_job(path):
        assert 'CELL03' not in path.name
        seen.append(path.name)
        return {'cell': 0, 'path': str(path)}
    monkeypatch.setattr(m, 'job', fake_job)
    data = m.inventory(tmp_path)
    assert data['fixed_denominator'] == 16 and data['rows_inspected'] == 15
    assert len(data['rows']) == 15 and all(r['index'] != 3 for r in data['rows'])
    assert len(data['rows'][0]['online_jobs']) == 2 and len(seen) == 2


def test_missing_receipt_not_promoted(tmp_path):
    assert m.numerical(tmp_path, 1)['state'] == 'missing_or_invalid'


def test_receipt_hash_and_cell_join_fail_closed(tmp_path):
    folder = tmp_path/'agent_e3_gd_host_cuda_20261029_v1_cell01_MAIN01'
    folder.mkdir()
    (folder/'report.json').write_text('{}')
    (folder/'completion.json').write_text('{}')
    receipt = tmp_path/'agent_e3_gd_host_cuda_20261029_v1_consume_cell01_MAIN01.json'
    data = {'cell': 1, 'status': 'control5_cell_native_GD_verified_NOT_admitted',
            'scientific_admission': False, 'closed_loop_complete': False,
            'report_sha256': m.digest(folder/'report.json'),
            'completion_sha256': m.digest(folder/'completion.json')}
    receipt.write_text(json.dumps(data))
    assert m.numerical(tmp_path, 1)['state'] == 'saved_receipt_joined'
    data['cell'] = 2
    receipt.write_text(json.dumps(data))
    assert m.numerical(tmp_path, 1)['state'] == 'missing_or_invalid'
    data['cell'] = 1
    receipt.write_text(json.dumps(data))
    (folder/'report.json').write_text('{"changed":true}')
    assert m.numerical(tmp_path, 1)['state'] == 'missing_or_invalid'


def test_corrupt_online_journal_rejected(tmp_path):
    path = tmp_path/'job'
    create_job(path)
    with (path/'events.jsonl').open('ab') as stream:
        stream.write(b'{}\n')
    with pytest.raises(ValueError):
        m.job(path)
