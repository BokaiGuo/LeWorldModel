"""NumPy native-buffer/temp-file fixtures, never production upstream consume."""
from copy import deepcopy
from pathlib import Path
import json
import runpy
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_secant_delivery_20261029_v1 as m

f = runpy.run_path(str(m.ROOT/'tests/test_agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1.py'))
RAM, Memory = f['RAM'], f['Memory']


def source_fixture():
    p = f['material']()
    p.update(schema=m.upstream.NAME+'.prepare', roster=m.roster(), sources={'/RAM/class.py': 'a'*64},
        expected_model_hash='RAM_model', policy='RAM_policy', cp5_tasks=dict(path='/RAM/cp5', sha256='b'*64), **m.upstream.FLAGS)
    classes = [dict(path='/RAM/class.py', sha256='a'*64)]
    r = dict(schema=m.upstream.NAME+'.report', status='four_groups_settled_NOT_delivery', errors=[], delivery=None,
        prepared=m.AUTH['prepared']['path'], prepare_sha256=m.AUTH['prepared']['sha256'], sources_verified=True,
        source_manifest_sha256=m.digest(p['sources']), model_before=dict(original_state_dict='RAM_model'),
        model_after=dict(original_state_dict='RAM_model'), classes_before=classes, classes_after=deepcopy(classes), rows=[], **m.upstream.FLAGS)
    receipt = dict(schema=m.upstream.NAME+'.consume', status='all4_actual_cp5_secant_groups_verified_NOT_delivery',
        prepared=m.AUTH['prepared']['path'], prepare_sha256=m.AUTH['prepared']['sha256'], report_sha256=m.AUTH['report']['sha256'],
        groups=4, candidate_denominator=16, arms=[], delivery=None, **m.upstream.FLAGS)
    p['source_groups'] = []
    for identity in m.roster():
        root, h = identity['root'], identity['horizon']; task, pro = p['tasks'][root], p['provenance'][root]
        original = dict(**identity, zero=dict(immediate=0., reach=10.)); p['source_groups'].append(original)
        result = dict(schema=m.upstream.NAME+'.group', status='all_four_scored', errors=[], delivery=None,
            task_sha256=m.digest(task), provenance_sha256=m.digest(pro), source_group_sha256=m.digest(original),
            source_closure_sha256=m.digest(p['sources']), root_context_role=m.ROLE, actual_cp5_start=True,
            rows=[], eligible_indices=[0, 1, 2, 3] if identity['slot'] == 0 else [], **m.upstream.FLAGS)
        for index in range(4):
            actions = np.full((5*h, 2), np.float32((index+1)/16)).tolist()
            ledger = m.core.Ledger(m.live.predict_axes(h)); ledger.scope = dict(phase='RAM_source_path')
            path = RAM(task, Memory()).replay(task['root_context'], actions, False, ledger)
            eligible = identity['slot'] == 0
            result['rows'].append(dict(index=index, status='complete', actions=actions, path=path,
                immediate=[2., 1., 1., 3.][index] if eligible else -1., reach=9. if eligible else 11.,
                eligible=eligible, params=np.arctanh(np.asarray(actions).reshape(-1)).tolist(), radius=.25))
        cost = m.upstream.caps(h); result['resource_receipt'] = dict(attempted=cost, completed=deepcopy(cost))
        r['rows'].append(dict(**identity, status='all_four_scored', result=result))
        receipt['arms'].append(dict(**identity, status='all_four_scored', complete=True, scored=4,
            eligible_indices=deepcopy(result['eligible_indices']), resources=deepcopy(result['resource_receipt']), delivery=None, **m.upstream.FLAGS))
    inherited = m.live.predict_axes(276); inherited['jacobian_API'] = 56
    r['accounting'] = dict(cumulative_attempted=inherited, cumulative_completed=deepcopy(inherited))
    receipt['accounting'] = deepcopy(r['accounting']); return p, r, receipt


@pytest.fixture(scope='module')
def material():
    p, r, c = source_fixture(); groups, inherited = m.source_material(p, r, c, deepcopy(c))
    return dict(p, schema=m.NAME+'.prepare', groups=groups, inherited_resources=inherited,
        profile=m.profile(), rules=m.rules(), shape_census=m.shape_census())


@pytest.fixture(scope='module')
def saved(material):
    store = Memory(); rows = m.execute(material, lambda t, s: RAM(t, s), store, permit_ram=True)
    return material, rows, store


def test_fixed_all4_and_gate_order(material):
    assert [len(g['eligible_order']) for g in material['groups']] == [4, 0, 0, 0]
    assert material['groups'][0]['eligible_order'] == [1, 2, 0, 3]
    assert [g['selected_index'] for g in material['groups']] == [1, None, None, None]
    assert sum(m.caps(g['horizon'])['predict_API'] for g in material['groups']) == 16
    assert sum(m.caps(g['horizon'])['predict_API'] for g in material['groups'] if g['selected_index'] is not None) == 2


@pytest.mark.parametrize('damage', ['receipt', 'schema', 'flag', 'source', 'model', 'class', 'order', 'missinggroup',
    'missingitem', 'gate', 'raw', 'path', 'resource', 'authority', 'newroot', 'fallback'])
def test_whole_upstream_mutations(damage):
    p, r, c = source_fixture(); rebuilt = deepcopy(c); result = r['rows'][0]['result']
    if damage == 'receipt': c['extra'] = 1
    elif damage == 'schema': c['schema'] = 'alias.consume'; rebuilt = deepcopy(c)
    elif damage == 'flag': c['P1_admission'] = True; rebuilt = deepcopy(c)
    elif damage == 'source': result['source_closure_sha256'] = 'd'*64
    elif damage == 'model': r['model_after'] = {}
    elif damage == 'class': r['classes_after'] = []
    elif damage == 'order': r['rows'].reverse()
    elif damage == 'missinggroup': c['arms'].pop(); rebuilt = deepcopy(c)
    elif damage == 'missingitem': result['rows'].pop()
    elif damage == 'gate': result['rows'][0]['eligible'] = False
    elif damage == 'raw': result['rows'][0]['actions'][0][0] = 1.01
    elif damage == 'path': result['rows'][0]['path'][0]['action_history'][0][0] += .125
    elif damage == 'resource': c['arms'][0]['resources']['attempted']['predict_API'] = 0; rebuilt = deepcopy(c)
    elif damage == 'authority': c['report_sha256'] = '0'*64; rebuilt = deepcopy(c)
    elif damage == 'newroot': r['rows'][0]['root'] = 2
    else: c['arms'][1]['delivery'] = {'fallback': True}; rebuilt = deepcopy(c)
    with pytest.raises((ValueError, KeyError)): m.source_material(p, r, c, rebuilt)


def test_saved_all4_actual_cost_and_deliveries(saved):
    p, rows, store = saved
    assert [r['status'] for r in rows] == ['model_delivery_verified', 'no_witness', 'no_witness', 'no_witness'], rows
    arms = m.validate_rows(p, rows, store.events, permit_fixture=True)
    assert len(arms) == 4 and arms[0]['model_delivery_verified'] is True
    for row in rows[1:]:
        assert row['result']['native_calls'] == [] and row['result']['delivery'] is None
        assert all(v['status'] == 'not_applicable_no_witness' for v in row['result']['verification'])
    r = rows[0]['result']; assert len(r['native_calls']) == 2
    calls = [c['transcript'][0]['request'] for c in r['native_calls']]
    assert [c['segmented'] for c in calls] == [False, True] and calls[0]['parent_context'] == calls[1]['parent_context']
    d = r['delivery']; assert d['candidate_index'] == 1 and d['raw_edge'] == p['groups'][0]['items'][1]['actions']
    assert d['commit_raw5'] == d['raw_edge'] and d['lookahead_raw'] == []
    assert all(d[k] is False for k in m.FLAGS) and d['suffix_in_plan'] is False
    cost = m.accounting(p, rows)
    assert cost['new_attempted'] == cost['new_completed'] == cost['new_caps'] == m.caps(1)
    assert cost['cumulative_completed']['predict_API'] == 278 and cost['cumulative_completed']['jacobian_API'] == 56


@pytest.mark.parametrize('h', [1, 3])
def test_H1_H3_independent_own_context_replays(material, h):
    g = deepcopy(material['groups'][0 if h == 1 else 1]); root = g['root']
    for row in g['items']: row.update(immediate=1., reach=9., eligible=True)
    g.update(eligible_order=[0, 1, 2, 3], selected_index=0)
    sink = Memory(); out = m.group(material['tasks'][root], material['provenance'][root], g,
        lambda t, s: RAM(t, s), sink, source_sha=m.digest(material['sources']), permit_ram=True)
    assert not out['errors'] and out['model_delivery_verified'], out['errors']
    assert out['resource_receipt']['attempted'] == m.caps(h)
    assert [v['segments'] for v in out['verification']] == [[5*h], [5] if h == 1 else [5, 10]]
    assert len(out['delivery']['lookahead_raw']) == 5*(h-1)
    m.validate_group(material['tasks'][root], material['provenance'][root], g, out, sink.events,
        source_sha=m.digest(material['sources']), permit_fixture=True)


@pytest.mark.parametrize('damage', ['order', 'index', 'raw', 'history', 'source', 'callcount', 'IO', 'event', 'seal', 'flag', 'cost'])
def test_saved_delivery_mutations(saved, damage):
    p, rows, store = deepcopy(saved); r = rows[0]['result']
    events = [e['payload'] for e in store.events if e['kind'] == 'group_event' and e['slot'] == 0]
    if damage == 'order': r['eligible_order'].reverse()
    elif damage == 'index': r['selected_index'] = 0
    elif damage == 'raw': r['delivery']['raw_edge'][0][0] += .01
    elif damage == 'history': r['verification'][1]['path'][0]['latent_history'][0][0] += 1
    elif damage == 'source': r['task_sha256'] = 'e'*64
    elif damage == 'callcount': r['native_calls'].pop()
    elif damage == 'IO': next(e for e in r['native_calls'][0]['transcript'] if e['kind'] == 'program_packet')['arrays']['raw']['sha256'] = 'e'*64
    elif damage == 'event': events.append(deepcopy(events[0]))
    elif damage == 'seal': events[:] = [e for e in events if e['payload']['kind'] != 'both_predictions_sealed']
    elif damage == 'flag': r['physical_release'] = True
    else: r['resource_receipt']['attempted']['predict_API'] += 1
    with pytest.raises((ValueError, KeyError)):
        m.validate_group(p['tasks'][0], p['provenance'][0], p['groups'][0], r, events,
            source_sha=m.digest(p['sources']), permit_fixture=True)


def test_failure_no_reselection(material):
    class Failing(RAM):
        def replay(self, *args): raise RuntimeError('fixed RAM failure')
    store = Memory(); rows = m.execute(material, lambda t, s: Failing(t, s), store, permit_ram=True)
    r = rows[0]['result']; assert len(rows) == 4 and rows[0]['status'] == 'partial_runtime_failure'
    assert r['selected_index'] == 1 and r['delivery'] is None and len(r['native_calls']) == 1
    assert r['resource_receipt']['attempted']['replay_calls'] == 1
    assert all(x['status'] == 'blocked_after_runtime_failure' and x['result'] is None for x in rows[1:])


def test_mismatch_both_sealed_before_rejection(material):
    g = deepcopy(material['groups'][0]); g['items'][g['selected_index']]['path'][0]['latent_history'][-1][0] += 1
    sink = Memory(); r = m.group(material['tasks'][0], material['provenance'][0], g, lambda t, s: RAM(t, s), sink,
        source_sha=m.digest(material['sources']), permit_ram=True)
    assert r['status'] == 'partial_runtime_failure' and len(r['native_calls']) == 2 and not r['model_delivery_verified']
    assert any(e['payload']['kind'] == 'both_predictions_sealed' for e in sink.events)
    assert r['selected_index'] == 1 and r['delivery'] is None


@pytest.fixture
def disk_saved(tmp_path, material, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path); (tmp_path/'results').mkdir()
    output = tmp_path/'results'/(m.NAME+'_RAM'); p = deepcopy(material)
    store = m.Store(output, m.profile()); store.put('binding.json', dict(prepared=str(tmp_path/'fixture.json'), prepare_sha256='a'*64))
    model = dict(original_state_dict=p['expected_model_hash']); classes = [dict(path='/RAM/class.py', sha256='a'*64)]
    for phase in ('before',):
        store.event(dict(kind='source_'+phase, sha256=m.digest(p['sources'])))
        store.event(dict(kind='model_'+phase, hashes=model, classes=classes))
        store.event(dict(kind='precision_'+phase, value=m.old.precision_contract()))
    rows = m.execute(p, lambda t, s: RAM(t, s), store, permit_ram=True)
    store.event(dict(kind='precision_after', value=m.old.precision_contract()))
    store.event(dict(kind='model_after', hashes=model, classes=classes))
    store.event(dict(kind='source_after', sha256=m.digest(p['sources']))); store.close()
    r = dict(schema=m.NAME+'.report', status='all4_secant_delivery_settled_NOT_physical', errors=[],
        prepared=str(tmp_path/'fixture.json'), prepare_sha256='a'*64, rows=rows, source_manifest_sha256=m.digest(p['sources']),
        sources_verified=True, model_before=model, model_after=deepcopy(model), classes_before=classes, classes_after=deepcopy(classes),
        host=dict(device='cuda:0', allocation_sum=4., visible_nvidia_nodes=['RAM_ONLY']), precision=[m.old.precision_contract()]*2,
        wall_seconds=1., CUDA_peak_bytes=[0, 0], journal_tail=store.tail, accounting=m.accounting(p, rows),
        artifact_sha256={n: m.sha(output/n) for n in ('binding.json', 'events.jsonl')}, **m.FLAGS)
    publication = m.publish_report(store, r, 'a'*64, 0., clock=lambda: 2.)
    assert publication['report_sha256'] == m.sha(output/'report.json')
    original_sha = m.sha
    monkeypatch.setattr(m, 'verify', lambda *args: p)
    monkeypatch.setattr(m, 'sha', lambda path: 'a'*64 if str(path) == '/RAM/class.py' else original_sha(path))
    return output, p, r, store


def test_real_Store_complete_reader(disk_saved):
    output, p, r, store = disk_saved
    result = m.consume(output, m.sha(output/'report.json'), r['prepared'], r['prepare_sha256'], m.sha(output/'completion.json'), _fixture=True)
    assert result['status'] == 'RAM_FIXTURE_ONLY_NOT_PRODUCTION' and '.RAM_FIXTURE.' in result['schema']
    assert len(result['arms']) == 4 and result['accounting']['new_completed']['predict_API'] == 2
    with pytest.raises(ValueError, match='no RAM production'):
        m.consume(output, m.sha(output/'report.json'), r['prepared'], r['prepare_sha256'], m.sha(output/'completion.json'))
    assert all(line == m.canonical(json.loads(line))+b'\n' for line in (output/'events.jsonl').read_bytes().splitlines(keepends=True))
    assert sum(x.stat().st_size for x in output.iterdir()) < m.shape_census()['working_upper_bytes']
    assert result['completion']['inclusive_wall_seconds'] > r['wall_seconds']
    assert result['serialized_census']['NPY_count'] == 0


@pytest.mark.parametrize('damage', ['externalsha', 'reportsha', 'prepare', 'wallabove', 'wallbefore',
    'status', 'artifactextra', 'artifactsha', 'bytes', 'science', 'missing'])
def test_completion_full_census_and_inclusive_wall_rejection(disk_saved, damage):
    output, p, r, _ = disk_saved; cp = output/'completion.json'; c = json.loads(cp.read_bytes())
    if damage == 'reportsha': c['report_sha256'] = 'e'*64
    elif damage == 'prepare': c['prepare_sha256'] = 'e'*64
    elif damage == 'wallabove': c['inclusive_wall_seconds'] = m.profile()['wall_seconds']+1
    elif damage == 'wallbefore': c['inclusive_wall_seconds'] = r['wall_seconds']-.1
    elif damage == 'status': c['status'] = 'publication_wall_failed'
    elif damage == 'artifactextra': c['artifact_sha256']['phantom.npy'] = 'e'*64
    elif damage == 'artifactsha': c['artifact_sha256']['events.jsonl'] = 'e'*64
    elif damage == 'bytes': c['bytes_before_completion'] += 1
    elif damage == 'science': c['P3_admission'] = True
    elif damage == 'missing': cp.rename(output/'missing_completion.json')
    if damage != 'missing': cp.write_bytes(m.canonical(c))
    external = '0'*64 if damage in ('externalsha', 'missing') else m.sha(cp)
    with pytest.raises((ValueError, KeyError)):
        m.consume(output, m.sha(output/'report.json'), r['prepared'], r['prepare_sha256'], external, _fixture=True)


def test_real_publication_after_report_fsync_hash_and_final_wall_failure(tmp_path, monkeypatch):
    store = m.Store(tmp_path/'publication', m.profile())
    store.put('binding.json', dict(test='RAM')); store.event(dict(kind='RAM_event')); store.close()
    hashed = []; original = m.sha
    def sha(path): hashed.append(Path(path).name); return original(path)
    monkeypatch.setattr(m, 'sha', sha)
    times = iter([m.profile()['wall_seconds']-.1, m.profile()['wall_seconds']+.1])
    def clock():
        assert hashed[:3] == ['binding.json', 'events.jsonl', 'report.json']
        return next(times)
    with pytest.raises(ValueError, match='durable failure retained'):
        m.publish_report(store, dict(prepared='/RAM', wall_seconds=0.), 'a'*64, 0., clock=clock)
    assert {x.name for x in store.output.iterdir()} == {'binding.json', 'events.jsonl', 'report.json', 'completion.json', 'completion_failure.json'}
    c = json.loads((store.output/'completion.json').read_bytes())
    failure = json.loads((store.output/'completion_failure.json').read_bytes())
    assert c['inclusive_wall_seconds'] > 0 and failure['elapsed_seconds'] > m.profile()['wall_seconds']
    assert c['report_sha256'] == original(store.output/'report.json')


@pytest.mark.parametrize('axis', ['metadata', 'blobs'])
def test_real_writer_census_quota_failclosed(tmp_path, axis):
    store = m.Store(tmp_path/'quota', m.profile()); c = m.shape_census()
    if axis == 'metadata': store.metadata_bytes = c['metadata_upper_bytes']
    else: store.persisted_blobs = c['max_persisted_blobs']+1
    with pytest.raises(ValueError, match='census cap'): store.event(dict(kind='not_written'))
    store.close(); assert (store.output/'events.jsonl').read_bytes() == b''


@pytest.mark.parametrize('damage', ['blank', 'space', 'hash', 'tail', 'missing', 'orphan', 'model', 'budget', 'science', 'file'])
def test_real_Store_reader_mutations(disk_saved, damage):
    output, p, r, store = disk_saved; journal = output/'events.jsonl'
    if damage == 'blank': journal.write_bytes(journal.read_bytes()+b'\n')
    elif damage == 'space': journal.write_bytes(journal.read_bytes().replace(b'{', b'{ ', 1))
    elif damage == 'hash': journal.write_bytes(journal.read_bytes().replace(b'delivery_selection', b'bad_selection', 1))
    elif damage == 'tail': r['journal_tail'] = 'd'*64
    elif damage == 'missing': r['rows'].pop()
    elif damage == 'orphan': r['rows'][0]['result']['native_calls'].append({})
    elif damage == 'model': r['model_after'] = {}
    elif damage == 'budget': r['accounting']['new_completed']['predict_API'] += 1
    elif damage == 'science': r['P1_admission'] = True
    else: (output/'extra').write_bytes(b'no orphan')
    r['artifact_sha256']['events.jsonl'] = m.sha(journal); (output/'report.json').write_bytes(m.canonical(r))
    completion = json.loads((output/'completion.json').read_bytes())
    completion['report_sha256'] = m.sha(output/'report.json')
    completion['artifact_sha256'] = {n: m.sha(output/n) for n in ('binding.json', 'events.jsonl', 'report.json')}
    completion['bytes_before_completion'] = sum((output/n).stat().st_size for n in completion['artifact_sha256'])
    (output/'completion.json').write_bytes(m.canonical(completion))
    with pytest.raises((ValueError, KeyError, json.JSONDecodeError)):
        m.consume(output, m.sha(output/'report.json'), r['prepared'], r['prepare_sha256'], m.sha(output/'completion.json'), _fixture=True)


def test_census_and_frozen_sources():
    c = m.shape_census(); assert c['blobs_per_packet'] == 19 and c['max_packets'] == 16
    assert c['native_payload_bytes'] == 413440
    assert c['working_upper_bytes'] < m.profile()['output_bytes']-m.profile()['final_reserve_bytes']
    assert all(m.sha(f) == h for f, h in m.PINS.items())


def test_signed_zero_native_bits_not_allclose(saved):
    call = deepcopy(saved[1][0]['result']['native_calls'][0]['transcript'])
    e = next(x for x in call if x['kind'] == 'program_packet')
    a = m.signal.unblob(e['arrays']['latent']).copy(); index = tuple(np.argwhere(a == 0)[0])
    a[index] = np.float32(-0.0); e['arrays']['latent'] = m.signal.blob(a)
    assert m.old.audit_native(call) is True  # old numeric equality accepts signed zero
    with pytest.raises(ValueError, match='native array exact'): m.bit_audit(call)


def test_all4_real_Store_full_shape_byte_census(tmp_path, material):
    p = deepcopy(material)
    for g in p['groups']:
        for row in g['items']: row.update(immediate=1., reach=9., eligible=True)
        g.update(eligible_order=[0, 1, 2, 3], selected_index=0)
    store = m.Store(tmp_path/'all4_RAM', m.profile())
    rows = m.execute(p, lambda t, s: RAM(t, s), store, permit_ram=True)
    assert all(r['status'] == 'model_delivery_verified' for r in rows), rows
    store.close(); store.put('report.json', dict(rows=rows))
    count = payload = 0
    def visit(value):
        nonlocal count, payload
        if type(value) is dict:
            if set(value) == {'shape', 'dtype', 'sha256', 'data'}:
                array = m.signal.unblob(value); count += 1; payload += array.nbytes
                assert len(m.canonical(value))-len(value['data']) <= 256
            else:
                for v in value.values(): visit(v)
        elif type(value) is list:
            for v in value: visit(v)
    for line in (store.output/'events.jsonl').read_bytes().splitlines(): visit(json.loads(line))
    visit(json.loads((store.output/'report.json').read_bytes()))
    census = m.shape_census()
    assert count == 3*census['max_packets']*census['blobs_per_packet']
    assert payload == 3*census['native_payload_bytes']
    assert sum(x.stat().st_size for x in store.output.iterdir()) < census['working_upper_bytes']
    assert m.accounting(p, rows)['new_completed']['predict_API'] == 16


def test_load_source_whole_consume_boundary(monkeypatch):
    p, r, c = source_fixture(); calls = []; rebuilt = deepcopy(c)
    refs = {m.AUTH[k]['path']: v for k, v in [('prepared', p), ('report', r), ('receipt', c)]}
    monkeypatch.setattr(m.upstream, 'read_pin', lambda ref, cap: deepcopy(refs[ref['path']]))
    def consume(*args): calls.append(args); return deepcopy(rebuilt)
    monkeypatch.setattr(m.upstream, 'consume', consume)
    monkeypatch.setattr(m, 'sha', lambda p: next(v['sha256'] for v in m.AUTH.values() if v['path'] == str(p)))
    _, groups, _ = m.load_source(); assert len(calls) == 1 and len(groups) == 4
    assert calls[0] == (Path(m.AUTH['report']['path']).parent, m.AUTH['report']['sha256'], m.AUTH['prepared']['path'], m.AUTH['prepared']['sha256'])
    c['unvalidated_extra'] = True
    with pytest.raises(ValueError, match='FULL persisted upstream receipt'): m.load_source()


def test_cold_cli_and_authority():
    with pytest.raises(ValueError): m.run('/absent', 'a'*64, '/absent', authorization='worker')
    run = subprocess.run([sys.executable, '-c', 'import sys;from experiments.spectral_reachability import agent_p3_secant_delivery_20261029_v1 as m;assert "torch" not in sys.modules;assert "pymunk" not in sys.modules;m.main()', '--help'], capture_output=True, text=True)
    assert run.returncode == 0 and all(x in run.stdout for x in ('prepare', 'verify', 'run', 'consume'))
