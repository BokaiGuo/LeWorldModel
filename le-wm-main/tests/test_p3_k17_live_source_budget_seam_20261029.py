import copy
from contextlib import nullcontext

import pytest
from experiments.spectral_reachability import agent_p3_c3_k17_native_meter_20261029_v1 as native_meter
from experiments.spectral_reachability import agent_p3_c3_k17_native_outer_20261029_v1 as outer
from experiments.spectral_reachability import agent_p3_c3_k17_native_store_20261029_v1 as native_store
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as journal
from experiments.spectral_reachability import p3_k17_live_source_budget_seam_20261029 as seam


def source():
    rows = []
    for slot, root, mode in outer.ORDER:
        rows.append(({'slot': slot, 'root': root, 'mode': mode, 'K': 17,
                      'first5_raw_f32_hex': bytes(range(40)).hex()}, {'task': root}, {'ref': root}))
    return {'source': 'fixture', 'roots': [{}, {}], 'source_plan': {}}, rows


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setattr(outer, 'material', source)
    bound = seam.SourceBoundStore.from_saved(native_store.NativeStore(tmp_path/'native'))
    bound.emit('run_started', denominator=6, order=[list(x) for x in outer.ORDER])
    met = native_meter.Meter(bound)
    met.where = {'branch': 0, 'slot': 3, 'root_slot': 0, 'mode': 'fixedH1'}
    met.call('constructor', lambda: object())
    return bound, met


def guard(bound):
    bound.emit('k17_guard', accepted=True, proof={'physical_dispatch_ready': False,
               'history': {'accepted': True}, 'action': {'actual_replay_tokens_exact': True}})


def transition(bound, met, action, step=1, terminated=False):
    bound.emit('returned_transition', **met.where, phase='plan', step=step,
               requested_bits=action, actual_bits=action, actual_dtype='<f4',
               terminated=terminated, truncated=False)


def test_actual_store_meter_nested_callbacks_and_action(live):
    bound, met = live
    met.call('public_step', lambda: met.call('physics_step', lambda: 17), phase='warmup', step=1, raw_bits='00')
    guard(bound)
    action = bound.actions[0][0]
    assert met.call('public_step', lambda: 42, phase='plan', step=1, raw_bits=action) == 42
    transition(bound, met, action)
    counts = bound.summary()['counts']
    assert counts['public_step'] == {'attempted': 2, 'completed': 2, 'failed': 0}
    assert counts['physics_step']['completed'] == 1
    assert not bound.summary()['production_authorized']
    events = journal.read_journal(bound.path/'journal.jsonl')
    assert events[0]['kind'] == 'source_budget_binding'
    assert [x['kind'] for x in events].index('k17_guard') < len(events)-1
    assert met.counts['public_step']['attempted'] == counts['public_step']['attempted']


@pytest.mark.parametrize('case', ['no_guard', 'wrong_action', 'changed_source'])
def test_denies_before_real_callback(live, case):
    bound, met = live
    if case != 'no_guard': guard(bound)
    if case == 'changed_source': bound.rows[0][1]['task'] = 99
    action = bound.actions[0][0] if case != 'wrong_action' else 'ff'*8
    invoked = []
    with pytest.raises(ValueError):
        met.call('public_step', lambda: invoked.append(True), phase='plan', step=1, raw_bits=action)
    assert not invoked and bound.fault


def test_failure_never_refunds_and_cleanup_can_run(live):
    bound, met = live
    def fail(): raise RuntimeError('native failure')
    with pytest.raises(RuntimeError): met.call('reset', fail)
    assert bound.counts['reset'] == {'attempted': 1, 'completed': 0, 'failed': 1}
    done = []
    met.call('close', lambda: done.append(True), cleanup=True)
    assert done == [True] and bound.counts['close']['completed'] == 1
    with pytest.raises(ValueError): met.call('reset', lambda: None)


def test_intent_write_failure_never_calls_callback(live, monkeypatch):
    bound, met = live
    original = bound.store.emit
    def fail(kind, **fields):
        if kind == 'operation_dispatch_intent': raise OSError('disk')
        return original(kind, **fields)
    monkeypatch.setattr(bound.store, 'emit', fail)
    invoked = []
    with pytest.raises(OSError): met.call('reset', lambda: invoked.append(True))
    assert not invoked and bound.fault and bound.counts['reset']['attempted'] == 1
    assert bound.pending


def test_actual_transition_mismatch_and_termination_stop(live):
    bound, met = live
    guard(bound)
    action = bound.actions[0][0]
    met.call('public_step', lambda: None, phase='plan', step=1, raw_bits=action)
    transition(bound, met, action, terminated=True)
    invoked = []
    with pytest.raises(ValueError):
        met.call('public_step', lambda: invoked.append(1), phase='plan', step=2, raw_bits=bound.actions[0][1])
    assert not invoked


def test_transition_action_mismatch(live):
    bound, met = live
    guard(bound)
    action = bound.actions[0][0]
    met.call('public_step', lambda: None, phase='plan', step=1, raw_bits=action)
    with pytest.raises(ValueError): transition(bound, met, 'ff'*8)


def test_wrong_receipt_and_stale_source_are_rejected(live):
    bound, met = live
    fake = {'index': 1, 'resource': 'reset', **met.where, 'status': 'returned'}
    with pytest.raises(ValueError): bound.emit('operation_returned', receipt=copy.deepcopy(fake))
    assert bound.fault


def test_independent_cap_blocks_pre_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(outer, 'material', source)
    caps = dict(native_meter.CAPS)
    caps['reset'] = 1
    monkeypatch.setattr(native_meter, 'CAPS', caps)
    bound = seam.SourceBoundStore.from_saved(native_store.NativeStore(tmp_path/'native'))
    bound.emit('run_started', denominator=6, order=[list(x) for x in outer.ORDER])
    met = native_meter.Meter(bound)
    met.where = {'branch': 0, 'slot': 3, 'root_slot': 0, 'mode': 'fixedH1'}
    met.call('constructor', lambda: None)
    met.call('reset', lambda: None)
    with pytest.raises(ValueError, match='resource cap'):
        bound.emit('operation_dispatch_intent', receipt={'index': 2, 'resource': 'reset',
                   **met.where, 'status': 'pending'})
    assert bound.counts['reset']['attempted'] == 1 and bound.fault


def test_seam_wall_blocks_callback(live):
    bound, met = live
    bound.started_at -= native_meter.WALL + 1
    invoked = []
    with pytest.raises(ValueError, match='wall limit'):
        met.call('reset', lambda: invoked.append(True))
    assert not invoked and bound.fault


def test_rejects_source_mutation_before_run(tmp_path, monkeypatch):
    monkeypatch.setattr(outer, 'material', source)
    bound = seam.SourceBoundStore.from_saved(native_store.NativeStore(tmp_path/'native'))
    bound.data['source'] = 'modified'
    with pytest.raises(ValueError, match='source changed'):
        bound.emit('run_started', denominator=6, order=[list(x) for x in outer.ORDER])


def test_existing_outer_integration_without_physics(tmp_path, monkeypatch):
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
    def episode(env, root, plan, item, task, ref, row, met, store):
        guard(store)
        actions = bytes.fromhex(item['first5_raw_f32_hex'])
        for step in range(1, 6):
            action = actions[(step-1)*8:step*8].hex()
            met.call('public_step', lambda: None, phase='plan', step=step, raw_bits=action)
            transition(store, met, action, step=step)
        return {'status': 'horizon_completed', 'returned': 5, 'missing_steps': []}
    monkeypatch.setattr(outer.episode, 'episode', episode)
    bound = seam.SourceBoundStore.from_saved(native_store.NativeStore(tmp_path/'native'))
    result = outer.execute_material(bound.data, bound.rows, bound, Env)
    assert not result['errors'] and len(result['rows']) == 6
    assert bound.counts['public_step']['completed'] == 30
    assert bound.counts['constructor']['completed'] == bound.counts['close']['completed'] == 6
    assert all(row['status'] == 'horizon_completed' for row in result['rows'])
