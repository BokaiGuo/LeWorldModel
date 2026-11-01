"""RAM fixtures and temp disk only; never invoke CUDA or physics."""
from pathlib import Path
import runpy
import time
import pytest
import test_agent_p1_two_packet_uphill_generator_20261029_v1 as source
import test_agent_p1_nonlinear_suffix_library_20261028_v2 as scalar_fixture
from experiments.spectral_reachability import agent_p1_joint_vreach_host_cuda_20261029_v1 as host

inputs = source.inputs
complete = source.complete
requests = source.requests


def test_contract_and_authorization():
    assert host.contract()['arm_denominator'] == 8
    assert host.contract()['candidate_denominator'] == 160
    with pytest.raises(ValueError, match='authorization'):
        host.run('x', 'y', 'z', 'wrong')


def test_prepare_binds_authority_and_detects_source_drift(tmp_path, monkeypatch, requests):
    (tmp_path/'results').mkdir()
    monkeypatch.setattr(host, 'ROOT', tmp_path)
    source_path = tmp_path/'authority.json'
    source_path.write_text('fixture authority')
    module_file = tmp_path/'owned.py'
    module_file.write_text('fixture source')
    monkeypatch.setattr(host, 'own_files', lambda: [module_file])
    a = dict(production_ready=False, P1_admission=False, physical_release=False,
        bindings={}, authority=dict(whole_producer_strict_reconstructed=True,
            rows=[dict(request=q) for q in requests], policy='fixture', expected_model_hash='a'*64))
    monkeypatch.setattr(host.authority, 'verify', lambda path, sha: a)
    prepare = host.build(str(source_path), host.sha(source_path))
    path = tmp_path/'results'/(host.NAME+'_prepare_fixture.json')
    receipt = host.write(path, prepare)
    assert host.verify(path, receipt['sha256']) == prepare
    with pytest.raises(ValueError): host.write(path, prepare)
    module_file.write_text('changed')
    with pytest.raises(ValueError): host.verify(path, receipt['sha256'])


def test_partial_persists_and_saved_consumer_audits_first_arm(tmp_path, monkeypatch, requests):
    (tmp_path/'results').mkdir()
    monkeypatch.setattr(host, 'ROOT', tmp_path)
    prepared = dict(requests=requests, bindings={}, expected_model_hash='a'*64)
    monkeypatch.setattr(host, 'verify', lambda *_: prepared)
    fixture = runpy.run_path(str(Path(__file__).with_name('test_agent_p3_parent_native_cuda_20261029_v1.py')))
    output = tmp_path/'results'/(host.NAME+'_fixture')
    store = host.Store(output, time.monotonic())
    store.put('binding.json', dict(prepared='/fixture/prepare', prepare_sha256='b'*64))
    rows = host.roster()
    def factory(q, journal, ledger):
        if q['slot'] == 0 and rows[1]['status'] == 'attempted':
            raise RuntimeError('fixture second arm failure')
        scalar = scalar_fixture.RAM(q['task'], journal, ledger)
        def component(task, sink):
            return host.component.Components.ram(root=task, caps=host.component.expected('postcontext_H1'),
                verification_reserve=dict.fromkeys(host.core.AXES, 0),
                factory=lambda t,s: fixture['NumpyNative'](t,s), sink=sink)
        return host.callback.NativeCallbacks(q['task'], scalar, component, journal.event)
    with pytest.raises(RuntimeError, match='second arm'):
        host.dispatch(prepared, store, rows, lambda: None, factory)
    assert [r['status'] for r in rows] == ['sealed', 'failed_partial'] + ['not_started']*6
    store.finalizing = True
    report = host.make_report(prepared, '/fixture/prepare', 'b'*64, store, rows,
        [{'stage': 'run', 'error': 'fixture'}], source_ok=False)
    store.put('report.json', report, final=True)
    result = host.consume(output, host.sha(output/'report.json'))
    assert result['status'] == 'partial_failclosed'
    assert result['audits'][0]['jacobian_VJP'] == 3840
    assert result['scalar_resources']['completed'] == host.F.caps(98,16)
    (output/'extra').write_bytes(b'x')
    with pytest.raises(ValueError, match='census'):
        host.consume(output, host.sha(output/'report.json'))


def test_failed_signal_prefix_is_durable_and_counted(tmp_path, monkeypatch, requests):
    (tmp_path/'results').mkdir(); monkeypatch.setattr(host, 'ROOT', tmp_path)
    p = dict(requests=requests, bindings={}, expected_model_hash='a'*64)
    monkeypatch.setattr(host, 'verify', lambda *_: p)
    fixture = runpy.run_path(str(Path(__file__).with_name('test_agent_p3_parent_native_cuda_20261029_v1.py')))
    output = tmp_path/'results'/(host.NAME+'_signal_failure')
    store = host.Store(output, time.monotonic())
    store.put('binding.json', dict(prepared='/fixture/prepare', prepare_sha256='b'*64))
    rows = host.roster()
    def factory(q, journal, ledger):
        scalar = scalar_fixture.RAM(q['task'], journal, ledger)
        def component(task, sink):
            return host.component.Components.ram(root=task, caps=host.component.expected('postcontext_H1'),
                verification_reserve=dict.fromkeys(host.core.AXES, 0),
                factory=lambda t,s: fixture['NumpyNative'](t,s,fail=True), sink=sink)
        return host.callback.NativeCallbacks(q['task'], scalar, component, journal.event)
    with pytest.raises(host.component.ComponentFailure):
        host.dispatch(p, store, rows, lambda: None, factory)
    store.finalizing = True
    report = host.make_report(p, '/fixture/prepare', 'b'*64, store, rows,
        [{'stage':'run','error':'signal failure'}], source_ok=False)
    store.put('report.json', report, final=True)
    result = host.consume(output, host.sha(output/'report.json'))
    assert result['signal_resources'][0]['attempted']['predict_API'] == 1
    assert result['signal_resources'][0]['returned_signals'] == 0
    assert result['rows'][0]['status'] == 'failed_partial'
    assert all(r['status'] == 'not_started' for r in result['rows'][1:])
