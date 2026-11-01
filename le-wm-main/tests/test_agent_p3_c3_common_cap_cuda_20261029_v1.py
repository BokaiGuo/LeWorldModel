"""RAM/native-buffer fixtures only. Never build/prepare/NN/physics/production strict."""
from copy import deepcopy
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_c3_common_cap_cuda_20261029_v1 as m

f = runpy.run_path(str(m.ROOT/'tests/test_agent_p3_r_native_orchestrator_20261029_v1.py'))
Native = f['fixtures']['NumpyNative']
Nonlinear = f['NonlinearRAM']
task = f['r_task']


class Memory(f['Memory']):
    def __init__(self): super().__init__(); self.results = {}
    def result(self, slot, value):
        self.results[slot] = deepcopy(value)
        return dict(name=f'slot_{slot:02}.json', sha256=m.digest(value))
    def result_reader(self, slot, ref):
        assert ref == dict(name=f'slot_{slot:02}.json', sha256=m.digest(self.results[slot]))
        return deepcopy(self.results[slot])


def one(mode='adaptive', K=17, cls=Nonlinear, t=None):
    t = task() if t is None else t; store = Memory()
    sink = type('Sink', (), {'event': lambda self, e: store.event(e)})()
    r = m.run_tree(t, lambda t, s: cls(t, s), sink, mode=mode, K=K, permit_ram=True, kind='RAM_FIXTURE',
        capsule=lambda cid, call: store.capsule(0, cid, call))
    return t, r, store


def validate(t, r, store):
    return m.validate_tree(t, r, store.events, permit_fixture=True,
        capsule_reader=lambda cid, call: store.reader(0, cid, call))


def test_fullshape_all27_axes_approved_caps_and_parent_bound():
    p = m.proof(); assert p['numerical_cost_bound_proved']
    assert p['max_parents'] == 17 and p['max_external_requests'] == 136 and p['identity_denominator'] == 144
    assert p['serialized_output_worstcase_fit_proved'] is False
    assert p['parent_J_shapes'] == [[192, 10], [192, 30]] and p['post_J_shape'] == [192, 10]
    for k in (1, 17):
        c = m.caps(k); assert set(c) == set(m.core.AXES) and len(c) == 27
        assert c['schedule_decisions'] == 17 and c['signal_compile_calls'] == c['jacobian_API']
        assert c['predict_context_tokens'] == c['pred_proj_rows'] == 3*c['predict_API']
        assert c['grad_API'] == c['raw_backward_callbacks'] == c['VJP_directions'] == 192*c['jacobian_API']
    assert m.caps(1)['replay_calls'] == 20 and m.caps(17)['replay_calls'] == 308
    assert m.profile()['output_bytes'] == 4*1024**3 and m.profile()['final_reserve_bytes'] == 512*1024**2
    assert m.profile()['wall_seconds'] == 7200 and m.profile()['slot_wall_seconds'] == 900
    assert len(m.roster()) == 12 and len({tuple(x.values()) for x in m.roster()}) == 12


@pytest.mark.parametrize('mode', m.MODES)
def test_policy_real_original_ABI_or_declared_static_counterfactual(mode):
    t = task(); root = m.parent.root_node(t); p = m.core.policy(t)
    for chi in (0., .6):
        for gain in (0., .1):
            s = m.core.FrozenSignals(p.source_seal, chi, gain)
            got = m.decision(t, root, s, mode)
            expected = m.core.choose(t, root, s, 'adaptive' if mode == 'adaptive' else 'fixed')
            if mode == 'staticH3_progress':
                expected.update(horizon=3, decision_source=m.NAME+'.staticH3_progress: declared static counterfactual')
            assert got == expected
            assert got['candidates'] == (8 if mode == 'adaptive' and chi >= .5 else 4)


@pytest.fixture(scope='module')
def R_saved(): return one()


def test_real_R_full0plus8_postJ_all_eligible_beam_not_onlywinner(R_saved):
    t, r, store = R_saved
    assert r['status'] == 'tree_complete_not_admitted', r['errors']
    g = r['R_groups'][0]; assert len(g['rows']) == 8 and g['control']['status'] == 'complete'
    assert len(g['eligible_indices']) > 1 and g['resources'] == m.tree.group_cost(1)
    elig = sorted(j['node']['indices'] for j in r['jobs'][:8] if j['status'] == 'returned_eligible')
    stopped = sorted(q['parent']['indices'] for q in r['queues'][1:])
    assert elig == stopped and all(q['status'] == 'stopped_by_original_child_policy' for q in r['queues'][1:])
    assert len(r['model_plan']['actions']) == 5 and len(r['jobs']) == 144
    assert all(x['status'] == 'returned' for x in r['replays'])
    for item in g['rows']:
        assert item['signal']['parent_context_sha256'] == m.digest(item['postcontext'])
        assert len(item['postcontext']['latent_history']) == 3 and len(item['postcontext']['action_history']) == 2
    s = validate(t, r, store); assert s['plan_available'] and s['AD_primal_predict_API'] > 0
    assert not any(r[k] for k in m.FLAGS)


def test_real_choose_R_H3_full30_chart_and_nine_full_postcontexts():
    t = f['fixtures']['original']['task']()
    # RAM task chosen analytically: short can move x only; long can move x,y,
    # and coordinate3 remains unreachable. Thus original chi>.5 and gain>.05.
    t['goal'] = [1., 1., 0., 2.]+[0.]*188; t['binding']['goal_sha256'] = m.digest(t['goal'])
    t, r, store = one(cls=Native, t=t)
    d = r['queues'][0]['decision']
    assert d['extension_type'] == 'reconfiguration' and d['horizon'] == 3 and d['candidates'] == 8
    g = r['R_groups'][0]; assert g['resources'] == m.tree.group_cost(3)
    assert g['resources']['predict_API'] == 54 and g['resources']['jacobian_API'] == 9
    assert all(len(x['actions']) == 15 and len(x['path']) == 3 and len(x['suffix_actions']) == 5
               and len(x['signal']['jacobian']) == 192 and len(x['signal']['jacobian'][0]) == 10
               for x in [g['control']]+g['rows'])
    assert len(r['signal_mappings'][1]['native_response']['jacobian'][0]) == 30
    validate(t, r, store)


@pytest.mark.parametrize('mode', m.MODES)
def test_K1_caps_retains_partial_without_best_plan_release(mode):
    t, r, store = one(mode=mode, K=1)
    assert r['status'] in ('resource_cap_before_signal', 'resource_cap_before_node_queue'), (r['status'], r['errors'])
    assert not r['errors'] and r['model_plan'] is None and r['diagnostic_best'] is not None
    assert all(v['status'] == 'not_requested' for v in r['replays'])
    assert r['resource_receipt']['attempted'] == r['resource_receipt']['completed']
    assert validate(t, r, store)['plan_available'] is False


@pytest.mark.parametrize('mode,h', [('fixedH1', 1), ('staticH3_progress', 3)])
def test_static_complete_native_three_layers_and_true_unchosen_panel_absence(mode, h):
    t, r, store = one(mode=mode, t=f['fixtures']['original']['task'](), cls=Native)
    assert r['status'] == 'tree_complete_not_admitted', r['errors']
    assert len(r['queues']) == 9 and r['realized_widths'] == [4]*9
    assert all(q['decision']['horizon'] == h and q['decision']['extension_type'] == 'progress' for q in r['queues'])
    assert r['R_groups'] == [] and max(j['node']['depth'] for j in r['jobs'] if j['node']) == 3
    # Best original model-Q may occur BEFORE depth3. Final replay pays the
    # actual selected chain, never pads to the longest-chain reserve.
    assert r['resource_receipt']['attempted']['predict_API'] == 72+36*h+2*(len(r['model_plan']['actions'])//5)
    assert r['resource_receipt']['attempted']['predict_API'] <= (114 if h == 1 else 198)
    assert len(r['native_calls']) == 18+36+2  # both J signals, only selected-H candidate calls.
    validate(t, r, store)


def test_full12_ordered_RAM_saved_strict_normal_cap_refusal_and_no_witness():
    # Complete real-dimensional callbacks. No production model/authority read.
    t = task(); u = task(); u['goal'] = [0.]*191+[1.]; u['binding']['goal_sha256'] = m.digest(u['goal'])
    p = dict(tasks=[t, u], profile=m.profile()); store = Memory()
    rows = m.execute(p, lambda t, s: Nonlinear(t, s), store, permit_ram=True)
    assert len(rows) == len(store.results) == 12
    assert all(r['status'] in m.SETTLED for r in rows), [(r['slot'], r['status']) for r in rows]
    assert any(r['status'] == 'no_witness' for r in rows)
    assert any(r['status'].startswith('resource_cap') for r in rows)
    assert any(r['status'] == 'tree_complete_not_admitted' for r in rows)
    summaries = m.validate_rows(p, rows, store.events, store.result_reader, store.reader, permit_fixture=True)
    assert len(summaries) == 12 and sum(len(r['jobs']) for r in store.results.values()) == 1728
    for root in (0, 1):
        for k in (1, 17):
            selected = [s['resources']['caps'] for s in summaries if s['root'] == root and s['K'] == k]
            assert len(selected) == 3 and selected[0] == selected[1] == selected[2]


@pytest.mark.parametrize('damage', ['unknown', 'orphan', 'missing', 'duplicate', 'J', 'raw', 'eligible', 'onlywinner', 'caps', 'mode', 'K', 'partialplan', 'sourceflag'])
def test_full_saved_arrays_events_decision_resource_refuse_mutation(R_saved, damage):
    t, original, s = R_saved; r = deepcopy(original); store = Memory(); store.events = deepcopy(s.events); store.capsules = deepcopy(s.capsules)
    if damage == 'unknown': store.events.append(dict(kind='io', call=None, payload=dict(kind='unknown_event')))
    elif damage == 'orphan': store.events[0]['call'] = 999
    elif damage == 'missing': store.events.pop(next(i for i, e in enumerate(store.events) if e['payload']['kind'] == 'direction'))
    elif damage == 'duplicate': store.events.append(deepcopy(store.events[0]))
    elif damage == 'J': r['R_groups'][0]['rows'][0]['signal']['jacobian'][0][0] += .125
    elif damage == 'raw': r['jobs'][0]['payload']['actions'][0][0] += .125
    elif damage == 'eligible': r['R_groups'][0]['rows'][0]['eligible'] = not r['R_groups'][0]['rows'][0]['eligible']
    elif damage == 'onlywinner': r['R_groups'][0]['eligible_indices'] = [r['R_groups'][0]['diagnostic_reach_winner']]
    elif damage == 'caps': r['resource_receipt']['caps']['predict_API'] -= 1
    elif damage == 'mode': r['mode'] = 'staticH3_progress'
    elif damage == 'K': r['K'] = 1
    elif damage == 'partialplan': r['status'] = 'resource_cap_before_node_queue'
    else: r['full_C3_demonstrated'] = True
    with pytest.raises((ValueError, AssertionError, KeyError)):
        validate(t, r, store)


def test_failure_keeps_nonrefunded_prefix_and_original12_denominator():
    p = dict(tasks=[task(), task()], profile=m.profile()); store = Memory()
    rows = m.execute(p, lambda t, s: Native(t, s, fail=True), store, permit_ram=True)
    assert rows[0]['status'] == 'failed_or_budget_exhausted'
    assert all(r['status'] == 'blocked_after_runtime_failure' and r['result'] is None for r in rows[1:])
    r = store.results[0]; assert r['resource_receipt']['attempted']['predict_API'] == 1
    assert r['resource_receipt']['completed']['predict_API'] == 0 and r['model_plan'] is None
    out = m.validate_rows(p, rows, store.events, store.result_reader, store.reader, permit_fixture=True)
    assert len(out) == 1


def test_explicit_slot_wall_failure_is_retained_not_relabelled_as_settled():
    p = dict(tasks=[task(), task()], profile=dict(m.profile(), slot_wall_seconds=-1)); store = Memory()
    rows = m.execute(p, lambda *args: pytest.fail('wall rejected before callback'), store, permit_ram=True)
    assert rows[0]['status'] == 'outer_runtime_failure' and all(r['status'] == 'blocked_after_runtime_failure' for r in rows[1:])
    with pytest.raises(ValueError): m.validate_rows(p, rows, store.events, store.result_reader, store.reader, permit_fixture=True)


def test_store_shards_exact_bytes_and_live_output_cap(tmp_path):
    store = m.Store(tmp_path/'RAM_new', m.profile())
    ref = store.result(0, dict(schema='RAM', literal=[1, 2, None]))
    assert ref['sha256'] == m.sha(store.output/ref['name'])
    assert store.bytes == (store.output/ref['name']).stat().st_size
    with pytest.raises(FileExistsError): store.result(0, {})
    store.limits = dict(store.limits, output_bytes=store.bytes+store.limits['final_reserve_bytes'])
    with pytest.raises(ValueError, match='output engineering cap'): store.result(1, {'too': 'large'})
    store.close()


def test_unknown_RAM_cannot_get_native_certification_and_cold_CLI(R_saved):
    t, r, s = R_saved
    with pytest.raises(ValueError, match='RAM production'): m.validate_tree(t, r, s.events, capsule_reader=lambda *a: None)
    with pytest.raises(ValueError, match='MAIN audit'): m.run('/not/read', 'bad', '/not/write', authorization='worker')
    assert all(m.sha(p) == h for p, h in m.PINS.items())
    code = f'import sys; import {m.__name__}; assert "torch" not in sys.modules and "gymnasium" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], cwd=m.ROOT, check=True)
    out = subprocess.run([sys.executable, '-m', m.__name__, '--help'], capture_output=True, text=True, cwd=m.ROOT)
    assert out.returncode == 0 and all(x in out.stdout for x in ('prepare', 'proof', 'verify', 'run', 'consume'))


@pytest.mark.parametrize('damage,match', [
    ('intact', 'ordered12'), ('extra', 'complete file census'), ('missing', 'complete file census'),
    ('bytes', 'all artifact bytes'), ('traversal', 'own exact artifact names'), ('binding', 'persisted binding'),
    ('hashchain', 'actual journal chain'), ('sourceflag', 'engineering closure'), ('oldC3alias', None)])
def test_saved_file_ABI_census_pins_hashchain_refuse(tmp_path, monkeypatch, damage, match):
    # Intentionally incomplete artifact fixture cannot receive any receipt;
    # only NEW own verify is stubbed to avoid production authority/build.
    output = tmp_path/'RAM_saved'; output.mkdir(); prepared = tmp_path/'RAM_prepare.json'; ps = 'a'*64
    p = dict(sources={'/RAM/class': 'b'*64}, expected_model_hash='RAM_model', profile=m.profile())
    monkeypatch.setattr(m, 'verify', lambda path, h: p)
    binding = dict(prepared=str(prepared), prepare_sha256=ps)
    (output/'binding.json').write_bytes(m.canonical(binding))
    for i in range(12): (output/f'slot_{i:02}.json').write_bytes(b'{}')
    model = dict(original_state_dict='RAM_model'); classes = [dict(path='/RAM/class', sha256='b'*64)]
    events = []
    for phase in ('before', 'after'):
        events += [dict(kind='model_'+phase, hashes=model, classes=classes),
                   dict(kind='source_'+phase, sha256=m.digest(p['sources'])),
                   dict(kind='precision_'+phase, value=m.old.precision_contract())]
    tail = '0'*64; lines = []
    for i, payload in enumerate(events):
        entry = dict(index=i, previous=tail, payload=payload); tail = m.digest(entry)
        lines.append(m.canonical(dict(entry, sha256=tail))+b'\n')
    (output/'events.jsonl').write_bytes(b''.join(lines))
    r = dict(schema=m.NAME+'.report', prepared=str(prepared), prepare_sha256=ps, errors=[],
        sources_verified=True, source_manifest_sha256=m.digest(p['sources']), model_before=model, model_after=model,
        classes_before=classes, classes_after=classes, precision=[m.old.precision_contract()]*2,
        host=dict(device='cuda:0', allocation_sum=4., visible_nvidia_nodes=['RAM_NOT_DEVICE']),
        artifact_sha256={x.name: m.sha(x) for x in output.iterdir()}, journal_tail=tail, rows=[], **m.FLAGS)
    if damage == 'extra': (output/'extra.json').write_bytes(b'{}')
    elif damage == 'missing': (output/'slot_11.json').unlink()
    elif damage == 'bytes': (output/'slot_00.json').write_bytes(b'{"changed":true}')
    elif damage == 'traversal': r['artifact_sha256']['../other.json'] = 'c'*64
    elif damage == 'binding':
        (output/'binding.json').write_bytes(m.canonical(dict(binding, prepare_sha256='c'*64)))
        r['artifact_sha256']['binding.json'] = m.sha(output/'binding.json')
    elif damage == 'hashchain':
        (output/'events.jsonl').write_bytes((output/'events.jsonl').read_bytes().replace(b'model_before', b'model_beforX'))
        r['artifact_sha256']['events.jsonl'] = m.sha(output/'events.jsonl')
    elif damage == 'sourceflag': r['full_C3_demonstrated'] = True
    elif damage == 'oldC3alias': del r['full_C3_demonstrated']; r['C3'] = False
    (output/'report.json').write_bytes(m.canonical(r))
    witness = dict(schema=m.NAME+'.completion', boundary='report_file_and_directory_fsync_returned',
        excludes='completion.json own serialization/write/fsync and subsequent process shutdown',
        report_sha256=m.sha(output/'report.json'), journal_tail=tail,
        wall_seconds_through_report_fsync=1., within_wall=True,
        artifacts={x.name: dict(sha256=m.sha(x), bytes=x.stat().st_size) for x in output.iterdir()})
    # Invalid-file mutations must still reach their original rejection, not be
    # hidden by a missing new completion witness.
    witness['artifacts'] = {n: witness['artifacts'].get(n, dict(sha256=h, bytes=0))
        for n, h in dict(r['artifact_sha256'], **{'report.json': m.sha(output/'report.json')}).items()}
    r['wall_seconds'] = 0.
    (output/'report.json').write_bytes(m.canonical(r))
    witness['report_sha256'] = m.sha(output/'report.json')
    witness['artifacts']['report.json'] = dict(sha256=witness['report_sha256'], bytes=(output/'report.json').stat().st_size)
    (output/'completion.json').write_bytes(m.canonical(witness))
    with pytest.raises((ValueError, KeyError), match=match): m.consume(output, m.sha(output/'report.json'), prepared, ps)


def test_byte_census_full12_repetitions_float_blob_and_hard_preload_refusal():
    b = m.byte_census(); assert len(b['slots']) == b['fixed_denominator'] == 12
    assert b['JSON_float_token_bytes_upper'] == 24 and b['native_blob_element_bytes_upper'] == 8
    assert b['live_allowance_bytes'] == 4*1024**3-512*1024**2
    assert b['live_bytes_upper'] == sum(x['bytes_upper'] for x in b['slots'])
    assert all(x['bytes_upper'] == sum(x['parts_bytes_upper'].values()) for x in b['slots'])
    for s in b['slots']:
        c = m.caps(s['K'])
        assert s['numeric_element_upper'] == c['predict_API']*8192+c['VJP_directions']*474+c['jacobian_API']*5952
        assert s['parts_bytes_upper']['capsules'] == 17*64*1024**2
        assert s['parts_bytes_upper']['shard'] == m.profile()['result_read_cap_bytes']
    assert not b['fits_live_allowance'] and b['live_bytes_upper'] > b['live_allowance_bytes']
    with pytest.raises(ValueError, match='preflight serialized upper bound'):
        m.run('/not/read', 'bad', '/not/create', authorization='MAIN_APPROVED_HOST_CUDA')


def test_live_writer_enforces_read_envelopes_before_persistence(tmp_path):
    store = m.Store(tmp_path/'envelope', dict(m.profile(), event_read_cap_bytes=8))
    with pytest.raises(ValueError, match='bounded complete event'): store.event(dict(kind='RAM'))
    assert store.bytes == 0 and store.index == 0
    store.close()


@pytest.mark.parametrize('damage', ['none', 'late', 'boundary', 'missing', 'bytes', 'report', 'oldwall', 'ownfsync'])
def test_completion_witness_after_actual_report_fsync_and_mutations(tmp_path, damage):
    store = m.Store(tmp_path/'completion', m.profile()); store.put('binding.json', {'RAM': True}); store.close()
    samples = iter([3., 5.]); store.started = 0.
    durable_put = store.put; put_returned = []
    def observe_put(name, value):
        durable_put(name, value); put_returned.append(name)
    store.put = observe_put
    def clock():
        assert put_returned == ['report.json']  # sample only AFTER durable put.
        return next(samples)
    store.clock = clock
    report = dict(wall_seconds=1., journal_tail=store.tail,
        artifact_sha256={x.name: m.sha(x) for x in store.output.iterdir()})
    w = m.persist_completion(store, report)
    assert w['wall_seconds_through_report_fsync'] == 3. and next(samples, None) is None
    assert 'completion.json' not in w['artifacts'] and 'report.json' in w['artifacts']
    if damage == 'none':
        assert m.validate_completion(store.output, report, m.sha(store.output/'report.json'), store.limits) == w
        return
    if damage == 'late': w['wall_seconds_through_report_fsync'] = 7201.
    elif damage == 'boundary': w['boundary'] = 'before_report_fsync'
    elif damage == 'missing': del w['artifacts']['events.jsonl']
    elif damage == 'bytes': w['artifacts']['report.json']['bytes'] += 1
    elif damage == 'report': w['report_sha256'] = 'f'*64
    elif damage == 'oldwall': report['wall_seconds'] = 4.
    else: w['excludes'] = 'nothing; own final fsync covered'
    (store.output/'completion.json').write_bytes(m.canonical(w))
    with pytest.raises(ValueError): m.validate_completion(store.output, report, m.sha(store.output/'report.json'), store.limits)


def test_completion_own_persistence_overrun_cannot_return_success(tmp_path):
    store = m.Store(tmp_path/'overrun', m.profile()); store.close(); store.started = 0.
    samples = iter([7199.75, 7200.5]); store.clock = lambda: next(samples)
    with pytest.raises(ValueError, match='completion persistence wall cap'):
        m.persist_completion(store, dict(journal_tail=store.tail, wall_seconds=7199.))
    assert (store.output/'completion.json').exists()  # failure never deletes evidence.
