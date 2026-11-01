"""RAM metadata/NumPy fixtures only: never execute a neural model or physics."""
from copy import deepcopy
import builtins
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_fixed_execution_cuda_20261028_v1 as m

spec = importlib.util.spec_from_file_location('fixed_cuda_ram', m.ROOT/'tests/agent_p3_fixed_execution_contract_20261028_v1_test.py')
ram = importlib.util.module_from_spec(spec); spec.loader.exec_module(ram)


@pytest.fixture(scope='module')
def full():
    panel, result, events = ram.complete()
    p = dict(panel=panel, profile=panel['profile'])
    r = dict(result, counts=m.counts(result), per_branch=m.fixed.audit_budget(panel, result['resource_receipt']),
        equal_planning_horizon=False, fixed_H1_cost_minimal_baseline=False, same_closed_loop_success_claim=False)
    initial = m.initial_result(panel, m.native.census(panel['base']['jobs']))
    return p, r, [dict(kind='denominator', rows=initial['rows'], prefixes=initial['prefixes'])]+events


def test_actual_frozen_panel_and_strict_saved_array_closure(full):
    p, r, events = full; m.validate_saved(p, r, events)
    assert r['counts'] == dict(task_methods=4, candidates=32, candidate_returns=32, selected_prefixes=4, prefix_returns=4)
    cost = r['resource_receipt']['completed']
    assert cost['predict_API'] == cost['predict_samples'] == 100
    assert cost['predict_context_tokens'] == 300 and cost['replay_calls'] == 68 and cost['metric_calls'] == 36
    assert cost['jacobian_API'] == cost['VJP_directions'] == cost['grad_API'] == 0
    assert cost['array_SVD_calls'] == cost['solve_calls'] == 4
    assert [b['planning_horizon_packets'] for b in r['per_branch']] == [1, 3, 1, 3]
    for b in r['per_branch']:
        assert b['signal_resources']['jacobian_API'] == 2 and b['signal_resources']['VJP_directions'] == 384
        assert b['total_resources']['predict_API'] == b['total_resources']['predict_samples'] == 33
        assert b['actual_physics_steps'] is b['CUDA_ns'] is b['FLOPs'] is None


@pytest.mark.parametrize('fault', ['raw', 'normalizer', 'token', 'latent', 'history', 'outgoing',
    'AE', 'predict', 'predictor', 'projection', 'endpoint', 'precision', 'missing_io', 'extra_buffer',
    'phase', 'step', 'orphan_phase', 'orphan_scope', 'moved_io', 'packet_operation',
    'actions', 'tail', 'request_endpoint', 'prefix_path', 'missing_prefix', 'prefix_order',
    'missing_candidate', 'winner', 'cost', 'denominator', 'queue', 'counts', 'API', 'samples',
    'free_J', 'prefix_resource', 'extra_event', 'physical_claim', 'cheap_fixed', 'error_suppressed'])
def test_mutations_failclosed(full, fault):
    p, r, events = deepcopy(full)
    prefix_events = [e for e in events if e.get('scope', {}).get('operation') == m.PREFIX]
    if fault in ('raw', 'normalizer', 'token', 'latent', 'history', 'outgoing', 'endpoint', 'precision', 'AE', 'predict', 'predictor', 'projection'):
        kind, key = {'normalizer': ('program_packet', 'std'), 'endpoint': ('program_boundary', 'latent'),
            'AE': ('io_return', 'action_encoder:output'), 'predict': ('io_return', 'predict:output'),
            'predictor': ('io_return', 'predictor:output'), 'projection': ('io_return', 'pred_proj:output'),
            'precision': ('program_packet', 'raw')}.get(fault, ('program_packet', fault))
        e = next(e for e in prefix_events if e['kind'] == kind and key in e['arrays'])
        a = m.signal.unblob(e['arrays'][key]).copy()
        if fault == 'precision': a = a.astype(np.float64)
        else: a.flat[0] += 1
        e['arrays'][key] = m.signal.blob(a)
    elif fault == 'missing_io': events.remove(next(e for e in prefix_events if e['kind'] == 'io_return'))
    elif fault == 'extra_buffer': next(e for e in prefix_events if e['kind'] == 'io_input')['arrays']['leaked_outcome'] = m.signal.blob(np.ones(1, np.float32))
    elif fault == 'phase': prefix_events[0]['phase'] = 'segmented'
    elif fault == 'step': prefix_events[0]['step'] = 1
    elif fault == 'orphan_phase':
        e = deepcopy(next(e for e in events if e['kind'] == 'program_packet')); e['phase'] = 'unused'; events.append(e)
    elif fault == 'orphan_scope': prefix_events[0]['scope']['child'] = True
    elif fault == 'moved_io':
        e = next(e for e in prefix_events if e['kind'] == 'io_return'); events.remove(e); events.append(e)
    elif fault == 'packet_operation': prefix_events[0]['operation_index'] = 0
    elif fault == 'actions': r['prefixes'][0]['request']['commands'][0][0] += 1
    elif fault == 'tail': r['prefixes'][1]['request']['nonexecuted_suffix_steps'] = 0
    elif fault == 'request_endpoint': r['prefixes'][0]['request']['predicted_execution_endpoint_sha256'] = 'a'*64
    elif fault == 'prefix_path': next(e for e in events if e['kind'] == 'selected_prefix_return')['path'] = []
    elif fault == 'missing_prefix': r['prefixes'].pop()
    elif fault == 'prefix_order': r['prefixes'].reverse()
    elif fault == 'missing_candidate': r['rows'].pop()
    elif fault == 'winner': r['winners'][0]['index'] = 999
    elif fault == 'cost': r['rows'][0]['cost'] += 1
    elif fault == 'denominator': events[0]['prefixes'].pop()
    elif fault == 'queue': p['panel']['queue'][0]['requested_execution_steps'] = 15
    elif fault == 'counts': r['counts']['candidates'] = 16
    elif fault in ('API', 'samples'): r['resource_receipt']['completed']['predict_'+fault] -= 1
    elif fault == 'free_J': p['panel']['base']['branches'][0]['signal_resources']['jacobian_API'] = 0
    elif fault == 'prefix_resource': r['resource_receipt']['events'][-1]['resource'] = 'replay'
    elif fault == 'extra_event': events.append(dict(kind='outcome', success=True))
    elif fault == 'physical_claim': r['equal_actual_physical_execution_horizon'] = True
    elif fault == 'cheap_fixed': r['fixed_H1_cost_minimal_baseline'] = True
    else: r['error'] = 'engineering fault'
    with pytest.raises((ValueError, KeyError, StopIteration)): m.validate_saved(p, r, events)


def disk_fixture(tmp_path, monkeypatch, full):
    p, r, events = deepcopy(full)
    policy = tmp_path/'policy'; policy.mkdir()
    for name in ('weights.pt', 'config.json'): (policy/name).write_bytes(b'RAM_METADATA_NOT_A_MODEL')
    p['panel']['base'].update(policy=str(policy), expected_model_hash='a'*64)
    p['sources'] = {str(policy/n): m.sha(policy/n) for n in ('weights.pt', 'config.json')}
    p['saved_verification'] = dict(run_upstream_array_SVD_calls=16, consume_upstream_array_SVD_calls=8,
        consume_generator_array_SVD_calls=4, native_new_array_SVD_calls=4,
        semantics='saved verification separate from native ledger; no neural CPU work')
    prepared = tmp_path/'prepare.json'; prepared.write_bytes(m.canonical(p)); expected = m.sha(prepared)
    # Replace only the NEW runner's upstream verifier for explicitly synthetic metadata.
    monkeypatch.setattr(m, 'verify', lambda path, h: deepcopy(p) if str(path) == str(prepared) and h == expected else m.require(False, 'RAM external prepare SHA'))
    model = dict(original_state_dict='a'*64, all_buffers='b'*64); weights = m.weight_hashes(p)
    r.update(schema=m.NAME+'.report', status='saved_model_prefix_not_admitted', prepared=str(prepared), prepare_sha256=expected,
        errors=[], model_before=model, model_after=model, weights_before=weights, weights_after=weights,
        sources_verified=True, source_manifest_sha256=m.digest(p['sources']),
        source_manifest_before=m.digest(p['sources']), source_manifest_after=m.digest(p['sources']),
        host=dict(device='cuda:0', allocation_sum=4.0, fixture='RAM ONLY, NOT HOST EVIDENCE'),
        saved_verification=p['saved_verification'], wall_seconds=1., CUDA_peak_bytes=[0, 0])
    events = events[:1]+[dict(kind='sources_before', sha256=m.digest(p['sources'])),
        dict(kind='weights_before', hashes=weights), dict(kind='model_before', hashes=model)]+events[1:]+[
        dict(kind='model_after', hashes=model), dict(kind='weights_after', hashes=weights),
        dict(kind='sources_final', sha256=m.digest(p['sources']))]
    output = tmp_path/'RAM_output'; store = m.signal.Store(output, p['profile'], free=lambda: 1024**3)
    store.put('binding.json', m.binding(prepared, expected, p))
    for e in events: store.event(e)
    store.close(); r.update(journal_tail=store.tail, artifact_sha256={n: m.sha(output/n) for n in ('binding.json', 'events.jsonl')})
    store.put('report.json', r)
    return p, r, output


def test_strict_saved_disk_roundtrip(tmp_path, monkeypatch, full):
    p, _, output = disk_fixture(tmp_path, monkeypatch, full)
    result = m.consume(output, m.sha(output/'report.json'))
    assert result['status'] == 'saved_prefix_verified_not_admitted'
    assert result['equal_requested_execution_horizon'] is True
    assert result['equal_actual_physical_execution_horizon'] is None
    assert all(result[k] is False for k in m.FLAGS)
    assert result['saved_verification_array_SVD_calls'] == 12
    assert result['prepare_sha256'] == m.sha(result['prepared'])
    assert result['root_contexts'] == [dict(slot=s, binding=t['binding'], context=t['root_context'])
        for s, t in enumerate(p['panel']['base']['tasks'])]


@pytest.mark.parametrize('fault', ['report_sha', 'prepare_sha', 'journal', 'binding', 'weights', 'artifact', 'extra_file', 'symlink',
    'source_after', 'model_after', 'partial', 'admission', 'wall', 'memory', 'host', 'saved_work'])
def test_persisted_closure_mutations(tmp_path, monkeypatch, full, fault):
    p, r, output = disk_fixture(tmp_path, monkeypatch, full)
    expected = m.sha(output/'report.json')
    if fault == 'report_sha': expected = '0'*64
    elif fault == 'journal':
        with (output/'events.jsonl').open('ab') as f: f.write(b'{}\n')
    elif fault == 'binding': (output/'binding.json').write_bytes(b'{}')
    elif fault == 'weights': (Path(p['panel']['base']['policy'])/'weights.pt').write_bytes(b'changed_RAM')
    elif fault == 'extra_file': (output/'unexpected').write_bytes(b'')
    elif fault == 'symlink': (output/'extra_link').symlink_to(output/'binding.json')
    else:
        if fault == 'prepare_sha': r['prepare_sha256'] = '0'*64
        elif fault == 'artifact': r['artifact_sha256']['events.jsonl'] = '0'*64
        elif fault == 'source_after': r['source_manifest_after'] = '0'*64
        elif fault == 'model_after': r['model_after'] = dict(r['model_after'], all_buffers='0'*64)
        elif fault == 'partial': r['status'] = 'partial_or_failed'
        elif fault == 'admission': r['P1_admission'] = True
        elif fault == 'wall': r['wall_seconds'] = 901.
        elif fault == 'memory': r['CUDA_peak_bytes'] = [2**40, 0]
        elif fault == 'host': r['host']['device'] = 'cpu'
        else: r['saved_verification'] = {}
        (output/'report.json').write_bytes(m.canonical(r)); expected = m.sha(output/'report.json')
    with pytest.raises((ValueError, KeyError)): m.consume(output, expected)


def test_authorization_rejected_before_any_new_model_import(monkeypatch):
    old = builtins.__import__; attempts = []
    def guarded(name, *a, **kw):
        if name.split('.')[0] in ('torch', 'gymnasium', 'pymunk'):
            attempts.append(name); raise AssertionError('forbidden model/physics import')
        return old(name, *a, **kw)
    monkeypatch.setattr(builtins, '__import__', guarded)
    with pytest.raises(ValueError, match='MAIN'): m.run(None, None, None, 'not_authorized')
    with pytest.raises(ValueError, match='MAIN'): m.load_backend(None, None, 'not_authorized')
    assert attempts == []


def test_loader_failure_preserves_full_denominator_without_dispatch(tmp_path, monkeypatch, full):
    p, _, _ = disk_fixture(tmp_path, monkeypatch, full)
    prepared = tmp_path/'prepare.json'; expected = m.sha(prepared)
    results = tmp_path/'results'; results.mkdir(); monkeypatch.setattr(m, 'ROOT', tmp_path)
    def unavailable(*args): raise RuntimeError('RAM injected unavailable HOST CUDA; no imports or execution')
    monkeypatch.setattr(m, 'load_backend', unavailable)
    output = results/(m.NAME+'_failed_RAM')
    with pytest.raises(ValueError, match='partial/failed'): m.run(prepared, expected, output, m.AUTH)
    r = m.signal.read(output/'report.json')
    assert r['status'] == 'partial_or_failed' and r['resource_receipt']['events'] == []
    assert len(r['rows']) == 32 and len(r['prefixes']) == 4
    assert all(row['status'] == 'blocked_not_dispatched' and row['actions'] is row['cost'] is None for row in r['rows'])
    assert all(x['status'] == 'blocked_not_dispatched' and x['request'] is None for x in r['prefixes'])
    assert r['model_before'] is r['model_after'] is r['per_branch'] is None
    assert r['counts']['candidate_returns'] == r['counts']['prefix_returns'] == 0
    with pytest.raises(ValueError, match='exclusive'): m.run(prepared, expected, output, m.AUTH)
    with pytest.raises(ValueError): m.consume(output, m.sha(output/'report.json'))


@pytest.mark.parametrize('mutated_final_model', [False, True])
def test_runner_end_to_end_control_flow_ram_only(tmp_path, monkeypatch, full, mutated_final_model):
    p, _, _ = disk_fixture(tmp_path, monkeypatch, full)
    prepared = tmp_path/'prepare.json'; expected = m.sha(prepared)
    results = tmp_path/'results'; results.mkdir(); monkeypatch.setattr(m, 'ROOT', tmp_path)
    # Rebind only names in NEW module, never frozen modules/globals. No torch,
    # neural model or physics object is constructed. HOST_CUDA here is mocked
    # metadata solely to exercise production dispatch/report control flow.
    class RamOnlyAdapter(ram.RamBackend):
        kind = 'HOST_CUDA'
        def __init__(self, model, device, task, store): super().__init__(task, store)
        def frozen(self): return None
    model = SimpleNamespace(modules=lambda: [])
    calls = []
    def model_hash(value):
        assert value is model; calls.append('hash')
        return dict(original_state_dict='a'*64, all_buffers=('c' if mutated_final_model and len(calls) == 2 else 'b')*64)
    monkeypatch.setattr(m, 'native', SimpleNamespace(**dict(vars(m.native), NativeProgram=RamOnlyAdapter)))
    monkeypatch.setattr(m, 'signal', SimpleNamespace(**dict(vars(m.signal), model_hash=model_hash)))
    def loader(p, store, authorization):
        assert authorization == m.AUTH; store.memory = lambda: (0, 0)
        fake_torch = SimpleNamespace(autocast=lambda **kw: nullcontext(), cuda=SimpleNamespace(synchronize=lambda device: None))
        return model, 'RAM_METADATA_ONLY', dict(device='cuda:0', allocation_sum=4.0, fixture='RAM'), fake_torch, lambda arg: nullcontext(), SimpleNamespace(MATH='RAM')
    monkeypatch.setattr(m, 'load_backend', loader)
    output = results/(m.NAME+'_complete_RAM')
    if mutated_final_model:
        with pytest.raises(ValueError, match='partial/failed'): m.run(prepared, expected, output, m.AUTH)
    else:
        receipt = m.run(prepared, expected, output, m.AUTH)
        assert m.consume(output, receipt['report_sha256'])['status'] == 'saved_prefix_verified_not_admitted'
    report = m.signal.read(output/'report.json')
    assert calls == ['hash', 'hash']
    assert report['counts']['candidate_returns'] == 32 and report['counts']['prefix_returns'] == 4
    assert report['resource_receipt']['completed']['predict_API'] == 100
    if mutated_final_model:
        assert report['status'] == 'partial_or_failed' and report['model_before'] != report['model_after']
        assert report['errors'][0]['stage'] == 'model_final'
        with pytest.raises(ValueError): m.consume(output, m.sha(output/'report.json'))


def test_prepare_verify_rebuild_in_new_module_only(tmp_path, monkeypatch, full):
    p = deepcopy(full[0]); p.update(contract_prepared='RAM.json', contract_prepare_sha256='a'*64)
    path = tmp_path/'p.json'; path.write_bytes(m.canonical(p)); expected = m.sha(path)
    monkeypatch.setattr(m, 'material', lambda path, expected: deepcopy(p))
    assert m.verify(path, expected) == p
    p['panel']['queue'][0]['requested_execution_steps'] = 15
    with pytest.raises(ValueError, match='rebuild'): m.verify(path, expected)


def test_exclusive_output_and_frozen_dependency_sha(tmp_path, monkeypatch):
    assert m.sha(m.fixed.__file__) == m.PANEL_SHA
    monkeypatch.setattr(m, 'ROOT', tmp_path); (tmp_path/'results').mkdir()
    candidate = tmp_path/'results'/(m.NAME+'_prepare_RAM.json')
    receipt = m.write_new(candidate, {'fixture': 'RAM metadata only'})
    assert receipt['sha256'] == m.sha(candidate)
    with pytest.raises(ValueError): m.write_new(candidate, {})
    with pytest.raises(ValueError): m.exclusive(tmp_path/'results'/'unrelated.json')


def test_cold_cli_no_torch_or_physics_import():
    script = 'import sys; from experiments.spectral_reachability import agent_p3_fixed_execution_cuda_20261028_v1 as m; m.main(["--help"])'
    proc = subprocess.run([sys.executable, '-B', '-c', script.replace('m.main(["--help"])',
        'assert not any(n.split(".")[0] in ("torch","gymnasium","pymunk") for n in sys.modules)')],
        cwd=m.ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
