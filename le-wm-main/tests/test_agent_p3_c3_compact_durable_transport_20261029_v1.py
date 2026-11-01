"""RAM callbacks and exclusive tmp_path disk only; never production strict."""
from copy import deepcopy
from pathlib import Path
import runpy
import struct
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p3_c3_compact_durable_transport_20261029_v1 as m


@pytest.fixture(scope='module')
def fx():
    return runpy.run_path(str(m.original.ROOT/'tests/test_agent_p3_c3_common_cap_cuda_20261029_v1.py'))


def setup_store(tasks, values):
    literals = []
    for r in values:
        if type(r) is not dict or 'R_groups' not in r: continue
        for group in r['R_groups']:
            b = group['bank']
            literals += [b['rule'], b['authority_manifest']['sources'], b['authority_manifest']['numeric_environment']]
    extra = dict(fields=sorted(m.codec.keys_of(values) | {'object_sha256', 'expanded_canonical_sha256'}), bank_literals=literals)
    seal = m.routing.seal_static(tasks, {}, extra)
    store, cat, budget = m.routing.install_static(seal)
    return m.native.NativeCatalog(store, cat), budget


def top(store, cat, event):
    encoder = m.native.CatalogEncoder(store, cat)
    raw, _ = encoder._encode_event(event); ref = store.intern(raw)
    index = len(store.events); stem = struct.pack('<Q32s32s', index, store.tail, ref)
    store.tail = m.codec.h(m.codec.CHAIN+stem)
    store.journal.append(stem+store.tail); store.events.append(ref); store.membership.append({'journal'})


@pytest.fixture(scope='module')
def one_fixture(fx):
    t = fx['task'](); execution = m.native.RamExecution()
    sink = type('Sink', (), {'event': lambda self, e: execution.event(e)})()
    r = m.original.run_tree(t, lambda t, s: fx['Nonlinear'](t, s), sink,
        mode='adaptive', K=17, permit_ram=True, kind='RAM_FIXTURE',
        capsule=lambda cid, call: execution.capsule(0, cid, call))
    return t, r, execution.events


def pack_single(one_fixture):
    t, r, events = one_fixture
    n, b = setup_store([t, deepcopy(t)], [r, events])
    n.ingest(0, t, r, events); n.finish_slot(0, r, events)
    return n, b


def test_integer_transport_census_not_fake_complete_fit():
    p = m.proof()
    assert p['base_conditional_live_bytes'] == 2930867648
    assert p['manifest_entries_bound'] == m.ROOTS*256+m.CALLS*1024+65536
    assert p['transport_live_upper'] == p['base_conditional_live_bytes']+sum(p['additions'].values())
    assert p['total_including_failure_reserve'] == p['transport_live_upper']+m.RESERVE
    assert p['remaining_live_bytes'] == m.codec.LIVE-p['transport_live_upper'] > 600*1024**2
    assert p['native_header_per_call_bound'] == m.CALL_HEADER_BOUND
    assert p['transport_additions_fit'] and not p['complete_production_fit_proved']
    assert p['output_cap'] == 4*1024**3 and p['failure_reserve'] == 512*1024**2
    assert p['missing'] and all(p[k] is False for k in m.FLAGS)


def test_fullshape_header_context_and_immutable_layout_task_no_generic_escape(fx):
    task = fx['task']()
    store, catalog, _ = m.routing.install_static(m.routing.seal_static([task, deepcopy(task)], {}, {}))
    h = m.Headers(store, catalog)
    context = deepcopy(task['root_context'])
    context['latent_history'] = [[float(i*192+j)/1024 for j in range(192)] for i in range(3)]
    context['action_history'] = [[-0.]+[float(i*10+j)/2048 for j in range(1,10)] for i in range(2)]
    before = h.new_bytes
    ref = h.put_header('context', context, m.CONTEXT_HEADER, m.CONTEXTS)
    assert h.new_bytes-before <= m.routing.bound(m.CONTEXT_HEADER)
    assert m.canonical(store.get(ref)) == m.canonical(context)
    for mutate in ['shape', 'type', 'extra']:
        bad = deepcopy(context)
        if mutate == 'shape': bad['latent_history'][0].pop()
        elif mutate == 'type': bad['action_history'][0][0] = 0
        else: bad['hidden_state'] = [0.]*192
        with pytest.raises(ValueError): h.put_header('context', bad, m.CONTEXT_HEADER, m.CONTEXTS)
    with pytest.raises(ValueError, match='count'):
        h.put_header('forbidden', context, m.CONTEXT_HEADER, 0)


@pytest.mark.parametrize('length', [5,15,45])
def test_replay_header_maxshape_with_exact_context_ref_and_signedzero(fx, length):
    task = fx['task'](); store, cat, _ = m.routing.install_static(
        m.routing.seal_static([task,deepcopy(task)], {}, dict(fields=['segmented'])))
    headers = m.Headers(store, cat)
    value = dict(parent_context=task['root_context'], actions=[[-0., 1.] for _ in range(length)], segmented=False)
    ref = headers.put_header('request', value, m.REPLAY_REQUEST, m.CALLS)
    assert m.canonical(store.get(ref)) == m.canonical(value)
    assert headers.new_bytes <= m.routing.bound(m.REPLAY_REQUEST)
    value['segmented'] = 0
    with pytest.raises(ValueError): headers.put_header('request', value, m.REPLAY_REQUEST, m.CALLS)


def test_online_append_dependencies_fsync_then_journal_and_exact_literals(tmp_path, fx):
    task = fx['task']()
    seal = m.routing.seal_static([task, deepcopy(task)], {}, dict(fields=['value']))
    store, _, _ = m.routing.install_static(seal)
    writer = m.Transport(tmp_path/'online', store)
    try:
        events = [dict(kind='precision_before', value=dict(value=v)) for v in [False, 0, 0., -0., True, 1, 1.]]
        for e in events: writer.append(e)
        assert len(writer.journal) == 7
        binding = writer.commit()
    finally: writer.close()
    reader = m.Reader(binding['path'], binding['manifest_sha256'])
    try:
        assert [m.canonical(reader.event(i)) for i in range(7)] == list(map(m.canonical, events))
        got = [reader.event(i)['value']['value'] for i in range(7)]
        assert list(map(type, got)) == [bool, int, float, float, bool, int, float]
        assert struct.pack('<d', got[3]) == struct.pack('<d', -0.)
        assert reader.disk.readonly and reader.journal.readonly
        with pytest.raises(ValueError): reader.disk[b'0'*32] = b'N'
    finally: reader.close()


def test_settled_R_whole_transcript_virtual_capsule_and_original_saved_validator(tmp_path, one_fixture):
    t, r, events = one_fixture; n, _ = pack_single(one_fixture)
    writer = m.import_settled(tmp_path/'settled', n.store)
    binding = writer.commit_result; writer.close()
    reader = m.Reader(binding['path'], binding['manifest_sha256'])
    try:
        expanded = [reader.event(i)['payload'] for i in range(len(reader.journal))]
        assert m.canonical(expanded) == m.canonical(events)
        saved = reader.store.get(reader.store.roots['slot_00'])
        assert m.canonical(saved) == m.canonical(r)
        out = m.original.validate_tree(t, saved, expanded, permit_fixture=True,
            capsule_reader=lambda cid, call: reader.capsule(0, cid, call))
        assert out['R_groups'] and out['plan_available']
        assert len(saved['jobs']) == 144 and len(out['resources']['caps']) == 27
        assert all(not Path(p).exists() for p in reader.store.capsules)
        assert reader.bytes == binding['artifacts_bytes']
    finally: reader.close()


def simple_disk(tmp_path, fx):
    t = fx['task']()
    seal = m.routing.seal_static([t, deepcopy(t)], {}, dict(fields=['value']))
    store, cat, _ = m.routing.install_static(seal)
    writer = m.Transport(tmp_path/'artifact', store)
    writer.append(dict(kind='precision_before', value=False))
    binding = writer.commit(); writer.close()
    return Path(binding['path']), binding['manifest_sha256']


@pytest.mark.parametrize('damage', ['manifest', 'static', 'pack', 'index', 'journal', 'completion', 'extra', 'missing', 'symlink'])
def test_complete_file_tamper_rejected(tmp_path, fx, damage):
    directory, sha = simple_disk(tmp_path, fx)
    names = dict(manifest='manifest.json', static='static.json', pack='cas/objects.pack',
                 index='cas/objects.idx', journal='journal.bin', completion='completion.json')
    if damage in names:
        p = directory/names[damage]
        with p.open('r+b') as f:
            f.seek(-1, 2); old = f.read(1); f.seek(-1, 2); f.write(bytes([old[0]^1]))
    elif damage == 'extra': (directory/'hidden').touch()
    elif damage == 'missing': (directory/'completion.json').rename(directory/'absent.json')
    else:
        (directory/'static.json').rename(tmp_path/'relocated.json')
        (directory/'static.json').symlink_to(tmp_path/'relocated.json')
    with pytest.raises((ValueError, UnicodeError, KeyError)): m.Reader(directory, sha)


def test_failed_append_retains_prefix_and_disallows_commit(tmp_path, fx):
    t = fx['task'](); store, _, _ = m.routing.install_static(m.routing.seal_static([t, deepcopy(t)], {}, {}))
    writer = m.Transport(tmp_path/'failed', store)
    try:
        writer.append(dict(kind='precision_before', value=False))
        with pytest.raises(ValueError): writer.append(dict(kind='unrecognised', arbitrary='payload'))
        assert writer.poisoned and len(writer.journal) == 1
        with pytest.raises(ValueError, match='unpoisoned'): writer.commit()
        assert not (writer.directory/'manifest.json').exists()
        assert (writer.directory/'journal.bin').stat().st_size == 104
        assert (writer.directory/'cas/objects.pack').stat().st_size > 0
    finally: writer.close()


def test_missing_completion_never_becomes_complete_and_no_overwrite(tmp_path, fx):
    directory, sha = simple_disk(tmp_path, fx)
    (directory/'completion.json').rename(tmp_path/'saved-completion')
    with pytest.raises(ValueError, match='census'): m.Reader(directory, sha)
    t = fx['task'](); store, _, _ = m.routing.install_static(m.routing.seal_static([t, deepcopy(t)], {}, {}))
    with pytest.raises(ValueError, match='exclusive'): m.Transport(directory, store)


def test_actual_shared_budget_preflight_is_not_two_independent_caps(tmp_path, fx):
    t = fx['task'](); store, _, _ = m.routing.install_static(m.routing.seal_static([t, deepcopy(t)], {}, {}))
    writer = m.Transport(tmp_path/'bounded', store)
    try:
        initial = writer.used()
        writer.extra_bytes = m.TOTAL-m.RESERVE-initial
        before = writer.disk.offset
        with pytest.raises(ValueError, match='shared live'): writer.store.intern(b'Snew unique allocated bytes')
        assert writer.disk.offset == before
    finally: writer.close()


def test_failure_reserve_retains_all_twelve_literal_statuses_not_success(tmp_path, fx):
    task = fx['task'](); store, _, _ = m.routing.install_static(m.routing.seal_static([task, deepcopy(task)], {}, {}))
    writer = m.Transport(tmp_path/'reserve', store)
    try:
        writer.append(dict(kind='precision_before', value=False))
        rows = [dict(**identity, status='not_requested', result=None) for identity in m.original.roster()]
        ref = writer.preserve_failure(rows, RuntimeError('untrimmed failure text\nwith literal -0'))
        saved = m.read_json(ref['path'], m.RESERVE)
        assert saved['rows'] == rows and len(saved['rows']) == 12
        assert saved['error']['message'].endswith('literal -0')
        assert saved['full_transport_verified'] is False and writer.poisoned
        with pytest.raises(ValueError): writer.commit()
        assert not (writer.directory/'manifest.json').exists()
    finally: writer.close()


def test_cli_has_no_production_prepare_run_or_fullstrict():
    r = subprocess.run([sys.executable, '-m', m.__name__, 'proof'], text=True, capture_output=True, check=True)
    assert 'complete_production_fit_proved":false' in r.stdout
    for cmd in ['prepare', 'run', 'consume']:
        r = subprocess.run([sys.executable, '-m', m.__name__, cmd], text=True, capture_output=True)
        assert r.returncode != 0


@pytest.mark.parametrize('fail', [True, False])
def test_online_original_twelve_loop_no_deferred_events(tmp_path, fx, fail):
    class Fail(fx['Native']):
        def __init__(self, task, sink): super().__init__(task, sink, fail=True)
    cls = Fail if fail else fx['Nonlinear']
    a = fx['task'](); b = deepcopy(a)
    b['goal'] = [0.]*191+[1.]; b['binding']['goal_sha256'] = m.original.digest(b['goal'])
    # Fixture field/literal discovery is NOT a production source seal. The
    # second invocation below is actual online event routing, not its replay.
    discovery = m.native.RamExecution()
    p, rows = discovery.run([a, b], lambda t, s: cls(t, s))
    n, _ = setup_store([a, b], list(discovery.results.values())+
        [discovery.events, rows, dict(prepared=p, rows=rows)])
    online = m.OnlineStore(tmp_path/'online-twelve', n.store, n.catalog, [a, b])
    try:
        rows, binding = online.run_fixture(lambda t, s: cls(t, s))
        assert len(rows) == 12
        assert len(online.store.events) == len(online.transport.journal)
        assert not online.transport.poisoned
        assert online.active is None
        census = online.routing.census()
    finally: online.transport.close()
    reader = m.Reader(binding['path'], binding['manifest_sha256'])
    try:
        p, saved = reader.fixture_inputs()
        events = [reader.event(i) for i in range(len(reader.journal))]
        summary = m.original.validate_rows(p, saved, events,
            reader.result, reader.capsule, permit_fixture=True)
        assert len(summary) == (1 if fail else 12)
        assert len(saved)*144 == 1728
        if fail:
            reconstructed = reader.reconstruct_fixture()
            assert len(reconstructed['all_rows']) == 12 and len(reconstructed['slots']) == 1
        print(m.canonical(dict(label='ONLINE_TEMP_DISK_RAM_ONLY', fail=fail,
            transport=reader.summary(), owner_events=dict(online.owner_events),
            routing=census)).decode())
    finally: reader.close()


@pytest.mark.parametrize('fail', [True, False])
def test_all12_slots_1728_identity_original_RAM_flow_through_disk_and_saved_rows(tmp_path, fx, fail):
    # Run the ORIGINAL complete12 loop. This is NOT fixture*12 byte evidence.
    class Fail(fx['Native']):
        def __init__(self, task, sink): super().__init__(task, sink, fail=True)
    cls = Fail if fail else fx['Nonlinear']
    a = fx['task'](); b = deepcopy(a)
    b['goal'] = [0.]*191+[1.]; b['binding']['goal_sha256'] = m.original.digest(b['goal'])
    execution = m.native.RamExecution()
    p, rows = execution.run([a, b], lambda t, s: cls(t, s))
    n, budget = setup_store([a, b], list(execution.results.values())+[execution.events, rows,
        dict(prepared=p, rows=rows)])
    refs = {}
    for e in execution.events:
        if e['kind'] == 'arm_event': continue
        if e['kind'] == 'arm_return':
            slot = e['row']['slot']; identity = m.original.roster()[slot]
            r = execution.results[slot]
            es = [x['payload'] for x in execution.events if x['kind'] == 'arm_event' and x['slot'] == slot]
            n.ingest(slot, p['tasks'][identity['root']], r, es)
            raw_ref = n.finish_slot(slot, r, es)['result']
            refs[slot] = {k: raw_ref[k] for k in ('slot', 'object_sha256', 'expanded_canonical_sha256')}
            e = deepcopy(e); e['row']['result'] = refs[slot]; rows[slot]['result'] = refs[slot]
        top(n.store, n.catalog, e)
    writer = m.import_settled(tmp_path/'twelve', n.store,
        fixture_execution=dict(prepared=p, rows=rows), catalog=n.catalog)
    binding = writer.commit_result; writer.close()
    reader = m.Reader(binding['path'], binding['manifest_sha256'])
    try:
        events = [reader.event(i) for i in range(len(reader.journal))]
        saved_rows = [e['row'] for e in events if e['kind'] == 'arm_return']
        assert len(rows) == 12 and len(m.original.roster())*144 == 1728
        if fail:
            assert len(saved_rows) == 1
            assert all(r['status'] == 'blocked_after_runtime_failure' for r in rows[1:])
        else: assert len(saved_rows) == 12
        saved_p, saved_all_rows = reader.fixture_inputs()
        assert m.canonical(saved_all_rows) == m.canonical(rows)
        summary = m.original.validate_rows(saved_p, saved_all_rows, events,
            reader.result, reader.capsule, permit_fixture=True)
        assert len(summary) == (1 if fail else 12)
        assert all(len(r['resources']['caps']) == 27 for r in summary)
        assert all(not r['full_C3_demonstrated'] for r in summary)
        print(m.canonical(dict(label='TEMP_DISK_RAM_ONLY', fail=fail, transport=reader.summary(),
                              static_budget=budget, completed_slots=len(summary))).decode())
    finally: reader.close()
