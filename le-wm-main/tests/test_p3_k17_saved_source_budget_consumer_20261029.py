import hashlib
import json
from contextlib import nullcontext

import pytest
from experiments.spectral_reachability import agent_p3_c3_k17_native_meter_20261029_v1 as meter
from experiments.spectral_reachability import agent_p3_c3_k17_native_outer_20261029_v1 as outer
from experiments.spectral_reachability import agent_p3_c3_k17_native_store_20261029_v1 as store
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport
from experiments.spectral_reachability import p3_k17_live_source_budget_seam_20261029 as seam
from experiments.spectral_reachability import p3_k17_saved_source_budget_consumer_20261029 as consumer


def source():
    rows = [({'slot': slot, 'root': root, 'mode': mode, 'K': 17,
              'first5_raw_f32_hex': bytes(range(40)).hex()}, {'task': root}, {'ref': root})
            for slot, root, mode in outer.ORDER]
    return {'roots': [{}, {}], 'source_plan': {}}, rows


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(outer, 'material', source)
    bound = seam.SourceBoundStore.from_saved(store.NativeStore(tmp_path/'native'))
    bound.emit('run_started', denominator=6, order=[list(x) for x in outer.ORDER])
    m = meter.Meter(bound)
    m.where = {'branch': 0, 'slot': 3, 'root_slot': 0, 'mode': 'fixedH1'}
    m.call('constructor', lambda: object())
    for step in range(1, 11):
        bits = bytes([step])*8
        m.call('public_step', lambda: None, phase='warmup', step=step, raw_bits=bits.hex())
        bound.emit('returned_transition', **m.where, phase='warmup', step=step, requested_bits=bits.hex(),
                   actual_bits=bits.hex(), actual_dtype='<f4', terminated=False, truncated=False)
    for index in range(36):
        name = f'fixture_{index}.png'
        content = f'fixture {index}'.encode()
        bound.store.inner.png(name, content)
        bound.emit('fresh_observation', receipt={'png': name, 'file_sha256': hashlib.sha256(content).hexdigest()})
    bound.emit('k17_guard', accepted=True, proof={'physical_dispatch_ready': False,
               'history': {'accepted': True}, 'action': {'actual_replay_tokens_exact': True}})
    action = bound.actions[0][0]
    m.call('public_step', lambda: None, phase='plan', step=1, raw_bits=action)
    bound.emit('returned_transition', **m.where, phase='plan', step=1, requested_bits=action,
               actual_bits=action, actual_dtype='<f4', terminated=False, truncated=False)
    return bound.path, m, bound


def audit(path, **kwargs):
    return consumer.consume(path, digest(path/'journal.jsonl'), consumer.sources(), **kwargs)


def rewrite(path, mutate):
    journal = path/'journal.jsonl'
    events = [json.loads(x) for x in journal.read_bytes().splitlines()]
    mutate(events)
    previous = None
    lines = []
    for index, event in enumerate(events):
        body = {'index': index, 'previous': previous, 'kind': event['kind'], 'payload': event['payload']}
        previous = hashlib.sha256(transport.canonical(body)).hexdigest()
        lines.append(transport.canonical(dict(body, sha256=previous)))
    journal.write_bytes(b'\n'.join(lines)+b'\n')


def test_partial_saved_report_is_durable_and_not_authority(recorded, tmp_path):
    path, _, _ = recorded
    out = tmp_path/'report.json'
    report = audit(path, output=out)
    assert json.loads(out.read_bytes()) == report
    assert report['active_branch'] == 0 and report['counts']['public_step']['completed'] == 11
    assert not any(report[k] for k in ('six_slot_event_coverage', 'complete_receipt_verified',
                                        'native_provenance_verified', 'production_authorized', 'scientific_pass'))
    with pytest.raises(FileExistsError): audit(path, output=out)


def test_external_anchors_required(recorded):
    path, _, _ = recorded
    with pytest.raises(ValueError, match='journal external SHA'):
        consumer.consume(path, '0'*64, consumer.sources())
    pins = consumer.sources()
    pins[next(iter(pins))] = '0'*64
    with pytest.raises(ValueError, match='external source SHA'):
        consumer.consume(path, digest(path/'journal.jsonl'), pins)


@pytest.mark.parametrize('target', ['binding', 'guard', 'action', 'receipt', 'missing_intent', 'missing_capture', 'false_close', 'unknown'])
def test_rechained_forgery_fails_semantics(recorded, target):
    path, _, _ = recorded
    def change(events):
        if target == 'binding': events[0]['payload']['source_sha256'] = '0'*64
        if target == 'guard': events[:] = [e for e in events if e['kind'] != 'k17_guard']
        if target == 'action':
            e = next(e for e in events if e['kind'] == 'operation_dispatch_intent' and e['payload']['receipt']['resource'] == 'public_step' and e['payload']['receipt']['phase'] == 'plan')
            e['payload']['receipt']['raw_bits'] = 'ff'*8
        if target == 'receipt':
            e = next(e for e in events if e['kind'] == 'operation_returned' and e['payload']['receipt']['resource'] == 'public_step' and e['payload']['receipt']['phase'] == 'plan')
            e['payload']['receipt']['step'] = 4
        if target == 'missing_intent':
            events[:] = [e for e in events if not (e['kind'] == 'operation_dispatch_intent' and e['payload']['receipt']['resource'] == 'public_step' and e['payload']['receipt']['phase'] == 'plan')]
        if target == 'missing_capture':
            events.pop(next(i for i, e in enumerate(events) if e['kind'] == 'fresh_observation'))
        if target == 'false_close':
            events.append({'kind': 'branch_closed', 'payload': {'row': {'index': 0, 'slot': 3, 'root_slot': 0,
                          'mode': 'fixedH1', 'status': 'horizon_completed'}}})
        if target == 'unknown': events.append({'kind': 'invented', 'payload': {}})
    rewrite(path, change)
    with pytest.raises(ValueError): audit(path)


def test_pending_and_recorded_failure_are_not_complete(recorded):
    path, m, _ = recorded
    def failure(): raise RuntimeError('fixture')
    with pytest.raises(RuntimeError): m.call('reset', failure)
    report = audit(path)
    assert report['fault_recorded'] and report['counts']['reset'] == {'attempted': 1, 'completed': 0, 'failed': 1}
    assert not report['complete_receipt_verified']
    rewrite(path, lambda events: events.pop())
    report = audit(path)
    assert report['pending'][-1]['resource'] == 'reset' and not report['complete_receipt_verified']


def test_missing_png_and_changed_source_rejected(recorded, monkeypatch):
    path, _, bound = recorded
    bound.emit('fresh_observation', receipt={'png': 'absent.png', 'file_sha256': '0'*64})
    with pytest.raises(ValueError, match='missing or symlink'): audit(path)
    monkeypatch.setattr(outer, 'material', lambda: ({'changed': True}, source()[1]))
    with pytest.raises(ValueError, match='source binding mismatch'): audit(path)


def test_six_slot_outer_fixture_coverage_is_not_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(outer, 'material', source)
    monkeypatch.setattr(outer.h3, 'validate_env', lambda body: None)
    monkeypatch.setattr(outer.h3, 'native_hooks', lambda *args: nullcontext())
    class Env:
        def __init__(self):
            self.unwrapped = self
            self.render_action = False
            self.render_size = 224
            self.render_mode = 'rgb_array'
        def close(self): pass
    def episode(env, root, plan, item, task, ref, row, met, bound):
        def observe(number):
            name = f'b{row["index"]}_{number}.png'
            content = name.encode()
            bound.store.inner.png(name, content)
            bound.emit('fresh_observation', receipt={'png': name, 'file_sha256': hashlib.sha256(content).hexdigest()})
        def step(phase, n, bits):
            met.call('public_step', lambda: None, phase=phase, step=n, raw_bits=bits)
            bound.emit('returned_transition', **met.where, phase=phase, step=n, requested_bits=bits,
                       actual_bits=bits, actual_dtype='<f4', terminated=False, truncated=False)
        for n in range(1, 11): step('warmup', n, (bytes([n])*8).hex())
        for n in range(36): observe(n)
        bound.emit('k17_guard', accepted=True, proof={'physical_dispatch_ready': False,
                   'history': {'accepted': True}, 'action': {'actual_replay_tokens_exact': True}})
        raw = bytes.fromhex(item['first5_raw_f32_hex'])
        for n in range(1, 6):
            step('plan', n, raw[(n-1)*8:n*8].hex())
            for j in range(3): observe(36+(n-1)*3+j)
        return {'status': 'horizon_completed', 'returned': 5, 'missing_steps': []}
    monkeypatch.setattr(outer.episode, 'episode', episode)
    bound = seam.SourceBoundStore.from_saved(store.NativeStore(tmp_path/'native'))
    result = outer.execute_material(bound.data, bound.rows, bound, Env)
    assert result['errors'] == []
    report = audit(bound.path)
    assert report['six_slot_event_coverage'] and report['counts']['public_step']['completed'] == 90
    assert not report['complete_receipt_verified'] and not report['scientific_pass']
