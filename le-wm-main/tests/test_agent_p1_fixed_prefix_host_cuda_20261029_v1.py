"""RAM and temporary disk only: no model, CUDA, physical execution or producer strict."""
from copy import deepcopy
import time
import pytest
from experiments.spectral_reachability import agent_p1_fixed_prefix_host_cuda_20261029_v1 as host
import test_agent_p1_fixed_prefix_saved_audit_20261029_v1 as fixture
import test_agent_p1_nonlinear_suffix_library_20261028_v2 as native

inputs = fixture.inputs
complete = fixture.complete
group = fixture.group
saved = fixture.saved


def groups(q):
    result = []
    for i in range(64):
        x = deepcopy(q)
        x.update(index=i, slot=i//16, source_arm=host.method.parent.ARMS[(i%16)//8], source_iteration=i%8+1)
        result.append(x)
    return result


def test_contract_fixed_paid_denominators():
    c = host.contract()
    assert c['group_denominator'] == 64 and c['candidate_denominator'] == 576
    assert c['new_total'] == host.F.caps(1216,512)
    assert c['inherited'] == host.F.caps(288,128)
    assert c['combined'] == host.F.caps(1504,640)
    assert c['FLOPs'] is None and not c['P1_admission'] and not c['physics']


def test_disk_saved_partial_consume_and_tamper(tmp_path, monkeypatch, saved):
    root = tmp_path; (root/'results').mkdir(); monkeypatch.setattr(host, 'ROOT', root)
    p = dict(groups=groups(saved[0]), bindings={}, expected_model_hash='a'*64)
    monkeypatch.setattr(host, 'verify', lambda *_: p)
    output = root/'results'/(host.NAME+'_fixture'); store = host.Store(output, time.monotonic())
    store.put('binding.json', dict(prepared='/fixture/prepare', prepare_sha256='b'*64))
    rows = host.roster(); serial = []
    def factory(task, journal, ledger):
        serial.append(1)
        if len(serial) == 2: raise RuntimeError('fixture failure')
        return native.RAM(task, journal, ledger)
    with pytest.raises(RuntimeError, match='fixture failure'):
        host.dispatch(p, store, rows, lambda: None, factory)
    assert rows[0]['status'] == 'sealed' and rows[1]['status'] == 'failed_partial'
    assert all(x['status'] == 'not_started' for x in rows[2:])
    store.finalizing = True
    artifacts = {str(f.relative_to(output)): host.sha(f) for f in output.rglob('*') if f.is_file()}
    report = dict(schema=host.NAME+'.report', status='partial_failclosed', contract=host.contract(),
        prepared='/fixture/prepare', prepare_sha256='b'*64, rows=rows, errors=[{'error':'fixture failure'}],
        fixed_denominator=64, candidate_denominator=576, artifact_sha256=artifacts,
        P1_admission=False, freshness=False, physical_release=False)
    store.put('report.json', report, final=True)
    receipt = host.consume(output, host.sha(output/'report.json'))
    assert receipt['status'] == 'partial_failclosed' and len(receipt['rows']) == 64
    assert receipt['actual_new_compute']['completed'] == host.F.caps(19,8)
    assert receipt['audits'][0]['scalar_objective']['checked_scalar_losses'] == 8
    (output/'unexpected').write_bytes(b'x')
    with pytest.raises(ValueError, match='file census'):
        host.consume(output, host.sha(output/'report.json'))


def test_wrong_authorization_rejected_before_verify(monkeypatch):
    monkeypatch.setattr(host, 'verify', lambda *_: pytest.fail('must not verify'))
    with pytest.raises(ValueError, match='authorization'):
        host.run('x', 'y', 'z', 'no')


def test_complete_64_disk_saved_consumer(tmp_path, monkeypatch, saved):
    root = tmp_path; (root/'results').mkdir(); monkeypatch.setattr(host, 'ROOT', root)
    p = dict(groups=groups(saved[0]), bindings={}, expected_model_hash='a'*64)
    monkeypatch.setattr(host, 'verify', lambda *_: p)
    output = root/'results'/(host.NAME+'_full_fixture'); store = host.Store(output, time.monotonic())
    store.put('binding.json', dict(prepared='/fixture/prepare', prepare_sha256='b'*64))
    rows = host.roster()
    host.dispatch(p, store, rows, lambda: None, lambda task, journal, ledger: native.RAM(task, journal, ledger))
    store.finalizing = True
    artifacts = {str(f.relative_to(output)): host.sha(f) for f in output.rglob('*') if f.is_file()}
    precision = dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False)
    model = dict(original_state_dict='a'*64)
    report = dict(schema=host.NAME+'.report', status='complete_model_only', contract=host.contract(),
        prepared='/fixture/prepare', prepare_sha256='b'*64, rows=rows, errors=[], fixed_denominator=64,
        candidate_denominator=576, artifact_sha256=artifacts, source_verified=True, source_manifest_sha256=host.digest({}),
        model_before=model, model_after=model, precision_before=precision, precision_after=precision,
        wall_seconds=1, CUDA_peak_bytes=[0,0], P1_admission=False, freshness=False, physical_release=False)
    store.put('report.json', report, final=True)
    receipt = host.consume(output, host.sha(output/'report.json'))
    assert receipt['complete'] and len(receipt['audits']) == 64
    assert receipt['actual_new_compute']['attempted'] == receipt['actual_new_compute']['completed'] == host.F.caps(1216,512)
