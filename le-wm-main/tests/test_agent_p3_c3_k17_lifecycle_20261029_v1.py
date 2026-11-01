import hashlib
import json
import pytest
from experiments.spectral_reachability import agent_p3_c3_k17_lifecycle_20261029_v1 as m


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def audit(path): return m.consume(path, sha(path/'report.json'), sha(path/'completion.json'))


def test_real_disk_six_and_early_stop(tmp_path):
    path = tmp_path/'run'
    def execute(identity, ctx):
        return {'status': 'native_stopped' if identity['index'] == 2 else 'horizon_completed',
                'returned': 2 if identity['index'] == 2 else 5}
    m.run(path, lambda: 'fixture', execute)
    receipt = audit(path)
    assert receipt['denominator'] == 6 and receipt['rows'][2]['status'] == 'native_stopped'
    assert not receipt['physical_provenance_verified']


def test_startup_failure_preserves_six(tmp_path):
    path = tmp_path/'run'
    def fail(): raise ValueError('fixture startup')
    called = []
    m.run(path, fail, lambda *x: called.append(x))
    result = audit(path)
    assert [r['status'] for r in result['rows']] == ['not_started']*6
    assert called == []


def test_slot_failure_retains_not_started(tmp_path):
    path = tmp_path/'run'
    def execute(identity, ctx):
        if identity['index'] == 2: raise RuntimeError('fixture')
        return {'status': 'horizon_completed'}
    m.run(path, lambda: None, execute)
    rows = audit(path)['rows']
    assert [x['status'] for x in rows] == ['horizon_completed']*2 + ['engineering_failed'] + ['not_started']*3


def test_tampering_and_missing_receipt_rejected(tmp_path):
    path = tmp_path/'run'
    m.run(path, lambda: None, lambda *x: {'status': 'horizon_completed'})
    report_sha, completion_sha = sha(path/'report.json'), sha(path/'completion.json')
    data = (path/'journal.jsonl').read_bytes()
    (path/'journal.jsonl').write_bytes(data.replace(b'fixedH1', b'fixedH2', 1))
    with pytest.raises(ValueError): m.consume(path, report_sha, completion_sha)
    (path/'completion.json').unlink()
    with pytest.raises(ValueError): m.consume(path, report_sha, completion_sha)


def test_store_failure_not_claimed_durable(tmp_path, monkeypatch):
    def fail(self, *args, **kwargs): raise OSError('disk failure')
    monkeypatch.setattr(m.bounded.Store, 'emit', fail)
    called = []
    with pytest.raises(OSError): m.run(tmp_path/'run', lambda: called.append(1), lambda *x: None)
    assert called == []
