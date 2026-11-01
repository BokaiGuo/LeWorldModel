"""RAM/native disk ABI only. No torch import, model loading, real prepare or fullstrict."""
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import time
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_two_program_postcontext_20261029_v1 as R

K, N = R.K, R.native


def ctx(root=0):
    z = np.random.default_rng(100+root).normal(0, .05, (3, 192)).astype(np.float32)
    return dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192,
                action_dim=10, latent_history=z.tolist(), action_history=np.zeros((2, 10), np.float32).tolist())


def packet(c, a):
    """Synthetic arithmetic callbacks with all original native shapes, not a neural model."""
    tokens = np.concatenate((np.asarray(c['action_history'], np.float32), N.kernel.normalized_history(a)))
    ae = np.tile(tokens, (1, 20))[:, :192].astype(np.float32)[None]
    z = np.asarray(c['latent_history'], np.float32)[None]
    pred = (z + np.float32(.03125)*ae).astype(np.float32)
    proj = pred.reshape(3, 192).copy()
    out = deepcopy(c)
    out.update(latent_history=np.concatenate((z[0, 1:], proj[-1:])).tolist(), action_history=tokens[1:].tolist())
    modules = [N.module_receipt(0, 'action_encoder', [N.blob(tokens[None])], N.blob(ae)),
               N.module_receipt(1, 'predictor', [N.blob(z), N.blob(ae)], N.blob(pred)),
               N.module_receipt(2, 'pred_proj', [N.blob(pred.reshape(3, 192))], N.blob(proj))]
    return out, modules


def path(c, u):
    cs = []
    for a in K.chart(u):
        c, _ = packet(c, a)
        cs.append(c)
    return cs


def slot(root=0):
    c = ctx(root)
    ps = []
    for candidate in (1, 2):
        u = np.random.default_rng(20+candidate+root).normal(0, .08, 30)
        ps.append(dict(id=f'r{root}/joint_H3_no_composition/c{candidate:02}/p0',
                       program_sha256=K.digest([root, candidate]), planned_steps=15, committed_steps=15,
                       logits=u.tolist(), contexts=path(c, u), input_context=deepcopy(c),
                       scope=dict(normalization=dict(mean=N.kernel.ACTION_MEAN.tolist(), std=N.kernel.ACTION_STD.tolist()))))
    return dict(root_slot=root, task_id=K.digest(['task', root]), task_sha256=K.digest(['tasksha', root]),
                root_context=c, root_warm_raw10=np.zeros((10, 2), np.float32).tolist(),
                goal=np.zeros(192).tolist(), programs=ps)


class FixtureBackend:
    def __init__(self, row, store):
        self.row, self.store = row, store
        self.event = lambda e: store.event(dict(slot=row['slot'], event=e))

    def rollout(self, c, actions, tag):
        cs = []
        for k, a in enumerate(actions):
            i = len(self.row['forwards'])
            request = N.packet_request(i, c, a, tag+f'/{k}')
            out, ms = packet(c, a)
            for j, m in enumerate(ms):
                m['index'] = 3*i+j
            f = dict(index=i, tag=tag+f'/{k}', request=request, input=deepcopy(c), raw=N.blob(a),
                     modules=ms, output=out, completed=True)
            self.event(dict(kind='forward_intent', request=request))
            self.event(dict(kind='forward_inputs', index=i, context=c, raw=f['raw']))
            for e in N.module_events(f):
                self.event(e)
            N.validate_packet(f)
            self.event(dict(kind='forward_return', index=i, context=out, completed=True))
            self.row['forwards'].append(f)
            cs.append(deepcopy(out))
            c = out
        return cs

    def jacobian(self, c, u, tag):
        i = len(self.row['AD'])
        request = R.ad_request(i, c, u, tag)
        self.event(dict(kind='AD_intent', request=request))
        paths = self.rollout(c, K.chart(u), tag+'/forward')
        # The synthetic J is not claimed as true neural or finite-difference evidence.
        j = np.zeros((192, 60), np.float32)
        buf = io.BytesIO()
        np.save(buf, j, allow_pickle=False)
        name = f'slot{self.row["slot"]}_J{i}.npy'
        self.store._write(name, buf.getvalue())
        d = dict(file=name, sha256=R.sha(self.store.output/name), shape=[192, 60], dtype='float32')
        self.row['AD'].append(dict(request=request, contexts=paths, array=d, completed=True))
        self.last_ad_contexts = deepcopy(paths)
        self.event(dict(kind='AD_return', index=i, array=d, contexts=paths, completed=True))
        return j.astype(float)


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    base = tmp_path_factory.mktemp('full30-fixture')
    output = base/(K.NAME+'_fixture')
    store = R.Store(output, time.monotonic())
    bindings = {}
    policy = base/'policy'
    policy.mkdir()
    loadfiles = []
    for name in ('config.json', 'weights.pt'):
        pth = policy/name
        with pth.open('xb') as f:
            f.write(b'fixture, not a model')
        h = R.sha(pth)
        bindings[str(pth)] = h
        d = dict(path=str(pth), sha256=h, bytes=pth.stat().st_size, hash_buffer_sha256=h,
                 deserialize_buffer_sha256=h, same_buffer=True)
        loadfiles.append(d)
        store.event(dict(kind='load_intent', path=str(pth), expected_sha256=h))
        store.event(dict(kind='consumed_buffer_frozen', receipt=d))
    source = str(Path(__file__).resolve())
    bindings[source] = R.sha(source)
    model_hash, prepared_sha = 'a'*64, 'b'*64
    prepared = str(base/'synthetic_prepare.json')
    p = dict(slots=[slot(0), slot(1)], policy=str(policy), expected_model_hash=model_hash, bindings_sha256=bindings,
             inherited_resources={'fixture': True, 'charged_again_as_new': False})
    load = dict(schema=N.loader.NAME+'.loader_receipt', prepare_sha256=prepared_sha,
                adapter_source_sha256=R.sha(N.loader.__file__),
                legacy_loader_source_sha256=N.loader.native.SOURCES['experiments/checkpoint_compat.py'],
                files=loadfiles, device='cuda:0', remapped_keys=[], missing=[], unexpected=[],
                tensor_hash_before=model_hash, frozen_eval=True)
    store.put('loader.json', load)
    store.put('classes.json', {source: bindings[source]})
    store.put('binding.json', dict(prepared=prepared, prepare_sha256=prepared_sha))
    summaries, rows = [], []
    for root in range(2):
        for ai, arm in enumerate(K.ARMS):
            i = 2*root+ai
            row = dict(slot=i, index=root, arm=arm, status='sealed', forwards=[], AD=[])
            store.event(dict(kind='slot_started', slot=i, root_slot=root, arm=arm))
            row['result'] = K.execute(p['slots'][root], arm, FixtureBackend(row, store))
            row['plan'] = R.row_plan(p, p['slots'][root], row)
            row['resources'] = K.resources(row['forwards'], row['AD'])
            row['memory'] = dict(allocated_peak=1, reserved_peak=1)
            store.put(f'slot{i}.json.gz', row)
            store.event(dict(kind='slot_sealed', slot=i, witness=row['plan']['witness_sha256']))
            summaries.append(dict(slot=i, root_slot=root, arm=arm, status='sealed', file=f'slot{i}.json.gz',
                                  wall_seconds=.1, final_Q=row['result']['final_Q'], contexts=row['result']['contexts'], resources=row['resources']))
            rows.append(row)
    store.close()
    store.put('seal.json', dict(schema=K.NAME+'.seal', complete=True, through_durable_data_seconds=.5,
              data_artifacts_sha256={f.name: R.sha(f) for f in output.iterdir()}), final=True)
    r = dict(schema=K.NAME+'.report', status='all4_full30_sealed_NOT_admitted', prepared=prepared,
             prepare_sha256=prepared_sha, contract=K.contract(), rows=summaries, errors=[], first_error=None,
             model_before=model_hash, model_after=model_hash, source_verified=True, host={'fixture': True},
             wall_seconds=1., actual_resources=R.census(R.journal(output)), inherited_resources=p['inherited_resources'],
             loader_memory=dict(allocated_peak=1, reserved_peak=1),
             artifact_sha256={f.name: R.sha(f) for f in output.iterdir()}, **K.FLAGS)
    store.put('report.json', r, final=True)
    return dict(output=output, p=p, report=r, rows=rows)


def test_full_four_slot_saved_native_consumer(bundle, monkeypatch):
    monkeypatch.setattr(R, 'verify', lambda *_: bundle['p'])
    result = R.consume(bundle['output'], R.sha(bundle['output']/'report.json'))
    assert len(result['plans']) == 4 and result['actual_new']['predict'] == 228
    assert result['P2_mechanism'] is False
    assert sum(len(r['forwards']) for r in bundle['rows']) == 228
    assert sum(len(r['AD']) for r in bundle['rows']) == 4


def test_real_store_singlelf_hashchain_and_exclusive_writer(tmp_path):
    out = tmp_path/(K.NAME+'_codec')
    s = R.Store(out, time.monotonic())
    s.event(dict(kind='a', request_index=4))
    s.event(dict(kind='b', context=ctx()))
    s.put('metadata.json', {'x': 1})
    s.close()
    assert R.journal(out) == [dict(kind='a', request_index=4), dict(kind='b', context=ctx())]
    data = gzip.decompress((out/'events.jsonl.gz').read_bytes())
    assert len(data.splitlines()) == 2 and b'\n\n' not in data
    with pytest.raises(FileExistsError):
        s.put('metadata.json', {'x': 2})
    with pytest.raises(ValueError):
        s.event({'kind': 'late'})


@pytest.mark.parametrize('mutation', ['nan', 'bool', 'shape', 'sphere'])
def test_domain_rejection(mutation):
    u = [0.]*60
    if mutation == 'nan': u[0] = float('nan')
    if mutation == 'bool': u[0] = True
    if mutation == 'shape': u.pop()
    if mutation == 'sphere': u[0] = K.C30+1
    with pytest.raises(ValueError):
        K.legal(u)


def test_raw_J_and_continuous_chain_formula():
    rng = np.random.default_rng(10)
    u = rng.normal(0, .1, 60)
    j = rng.normal(size=(192, 60)).astype(np.float32)
    z, goal = rng.normal(size=(2, 192))
    p = K.proposals(u, z, j, goal)
    expected = (1-np.tanh(u)**2)*(j.astype(float).T@(z-goal))*(2/192)
    np.testing.assert_allclose(p['gradient'], expected, rtol=1e-14, atol=1e-14)
    assert len(p['logits']) == 5
    assert all(np.linalg.norm(v) <= K.C30 for v in p['logits'])


def test_zero_gradient_keeps_all_four_literal_proposals():
    u = np.arange(60)/100
    p = K.proposals(u, np.zeros(192), np.zeros((192, 60)), np.zeros(192))
    assert p['logits'] == [u.tolist()]*5


def test_full_context_not_terminal_only():
    c = ctx()
    c2 = deepcopy(c)
    c2['latent_history'][0][0] += .25
    with pytest.raises(ValueError):
        K.equal(c, c2, 'complete context')


def test_new_B_is_first_postcontext_not_old_root(bundle):
    row = bundle['rows'][0]
    out = row['result']
    assert out['new_B_input_context'] == out['contexts'][2]
    assert out['new_B_input_context'] != out['inherited']['B_old_input']
    assert out['contexts'][3:] != out['inherited']['B_old_contexts']
    assert len(row['forwards']) == 18


def test_joint_stable_tie_and_independent_same_action(bundle):
    for i in (1, 3):
        r = bundle['rows'][i]['result']
        assert [x['selected'] for x in r['rounds']] == [0, 0]
        assert r['initial_contexts'] == bundle['rows'][i-1]['result']['contexts']
        assert r['final_Q'] <= r['initial_Q']
        assert len(bundle['rows'][i]['forwards']) == 96


@pytest.mark.parametrize('mutation', ['completed', 'dimension', 'coordinates', 'unknown', 'missing', 'splice'])
def test_saved_AD_and_complete_chronology_mutations(bundle, mutation):
    row = deepcopy(bundle['rows'][1])
    if mutation == 'completed': row['AD'][0]['completed'] = False
    if mutation == 'dimension': row['AD'][0]['request']['shape'] = [192, 30]
    if mutation == 'coordinates': row['AD'][0]['request']['coordinates'] = 'logit_zero'
    if mutation == 'unknown': row['forwards'][0]['unknown'] = 1
    if mutation == 'missing': row['forwards'].pop()
    if mutation == 'splice': row['forwards'][0], row['forwards'][1] = row['forwards'][1], row['forwards'][0]
    b = R.SavedBackend(row, lambda _: None, lambda _: np.zeros((192, 60), np.float32))
    with pytest.raises((ValueError, IndexError)):
        K.execute(bundle['p']['slots'][0], K.ARMS[1], b)
        b.complete()


def test_unknown_and_moved_event_rejected(bundle, monkeypatch):
    monkeypatch.setattr(R, 'verify', lambda *_: bundle['p'])
    real = R.journal(bundle['output'])
    for events in ([{'kind': 'unknown'}]+real, real[:10]+[real[11], real[10]]+real[12:]):
        monkeypatch.setattr(R, 'journal', lambda _, events=events: events)
        with pytest.raises(ValueError):
            R.consume(bundle['output'], R.sha(bundle['output']/'report.json'))


def test_native_module_coherent_corruption_rejected(bundle):
    row = deepcopy(bundle['rows'][0])
    blob = row['forwards'][0]['modules'][1]['inputs'][0]
    a = N.unblob(blob).copy()
    a.flat[0] += np.float32(.25)
    row['forwards'][0]['modules'][1]['inputs'][0] = N.blob(a)
    with pytest.raises(ValueError):
        K.execute(bundle['p']['slots'][0], K.ARMS[0], R.SavedBackend(row, lambda _: None, None))


def test_model_source_binding_mutation(tmp_path):
    p = tmp_path/'source.py'
    with p.open('xb') as f: f.write(b'a')
    h = R.sha(p)
    R.rehash({str(p): h})
    with p.open('wb') as f: f.write(b'b')
    with pytest.raises(ValueError): R.rehash({str(p): h})


def test_other_link_SHA_rejected_before_any_reader():
    with pytest.raises(ValueError, match='pinned MAIN linker'):
        R.build(R.LINK['path'], '0'*64)


def test_verify_exhausted_budget_cannot_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'verify', lambda *_: dict(expected_model_hash='a'*64, bindings_sha256={}, inherited_resources={}))
    times = iter([0., 1801.])
    monkeypatch.setattr(R.time, 'monotonic', lambda: next(times, 1802.))
    out = tmp_path/(K.NAME+'_expired')
    with pytest.raises(ValueError):
        R.run_cuda('fixture', 'f'*64, out, R.AUTH)
    r = json.loads((out/'report.json').read_bytes())
    assert r['status'] == 'failed_partial' and r['first_error'] is not None
    assert not (out/'loader.json').exists()
    assert r['actual_resources']['forward_attempted'] == 0


def test_wrong_authorization_no_IO(tmp_path):
    out = tmp_path/(K.NAME+'_noauth')
    with pytest.raises(ValueError): R.run_cuda('missing', 'f'*64, out, 'no')
    assert not out.exists()


def test_failed_nested_intents_not_refunded():
    events = [dict(slot=0, event=dict(kind='AD_intent')),
              dict(slot=0, event=dict(kind='forward_intent')),
              dict(slot=0, event=dict(kind='module_intent', resource='action_encoder'))]
    c = R.census(events)
    assert c['jacobian_attempted'] == c['forward_attempted'] == 1
    assert c['forward_returned'] == c['jacobian_returned'] == 0
    assert c['modules_attempted']['action_encoder'] == 1


def test_full_shapes_uncompressed_engineering_budget(bundle):
    """Count raw serialized full-width IO + ledger + result + arrays, not optimistic gzip."""
    events = R.journal(bundle['output'])
    budget = R.size_budget(bundle['rows'], events)
    print('FULL_SHAPE_BUDGET '+json.dumps(budget, sort_keys=True))
    assert budget['conservative_uncompressed_total'] < R.CAP-R.RESERVE
    assert all(x['array']['shape'] == [192, 60] for row in bundle['rows'] for x in row['AD'])


@pytest.mark.parametrize('field,value', [('wall_seconds', 1801.), ('wall_seconds', True), ('P2_mechanism', True)])
def test_overwall_bool_or_science_report_rejected(bundle, monkeypatch, field, value):
    monkeypatch.setattr(R, 'verify', lambda *_: bundle['p'])
    real = R.read
    report = dict(bundle['report'], **{field: value})
    monkeypatch.setattr(R, 'read', lambda path, *a, **kw: report if Path(path).name == 'report.json' else real(path, *a, **kw))
    with pytest.raises(ValueError): R.consume(bundle['output'], R.sha(bundle['output']/'report.json'))


@pytest.mark.parametrize('mutation', ['blank', 'reorder', 'hash', 'extra_gzip'])
def test_real_disk_journal_rejects_malformed_chain(tmp_path, mutation):
    out = tmp_path/(K.NAME+'_bad_chain')
    s = R.Store(out, time.monotonic())
    s.event({'kind': 'one'})
    s.event({'kind': 'two'})
    s.close()
    p = out/'events.jsonl.gz'
    lines = gzip.decompress(p.read_bytes()).splitlines(keepends=True)
    if mutation == 'blank': lines.insert(1, b'\n')
    if mutation == 'reorder': lines.reverse()
    if mutation == 'hash':
        e = json.loads(lines[0]); e['event_sha256'] = 'f'*64; lines[0] = K.raw(e)
    data = gzip.compress(b''.join(lines), mtime=0)
    if mutation == 'extra_gzip': data += gzip.compress(b'', mtime=0)
    with p.open('wb') as f: f.write(data)
    with pytest.raises(ValueError): R.journal(out)


def test_installed_metadata_sources_no_neural_import():
    import sys
    before = set(sys.modules)
    s = R.installed_sources()
    assert len(s) == 8
    assert not ({'torch', 'transformers'} & (set(sys.modules)-before))


def test_raw60_chart_is_literal_concat_not_f32_tanh_replacement():
    u = np.random.default_rng(19).normal(0, .25, 60)
    raw = K.chart(u)
    assert raw.dtype == np.float32 and raw.shape == (6, 5, 2)
    assert raw.tobytes() == K.chart(u[:30]).tobytes()+K.chart(u[30:]).tobytes()
    assert raw.tobytes() != np.tanh(u.astype(np.float32)).astype(np.float32).tobytes()


@pytest.mark.parametrize('bad', [np.bool_(True), True, '0.0', float('inf')])
def test_strict_numerical_runtime_inputs(bad):
    u = [0.]*60; u[0] = bad
    with pytest.raises(ValueError): K.legal(u)


def test_program_count_not_packet_count(bundle):
    assert [r['plan']['program_count'] for r in bundle['rows']] == [2, 1, 2, 1]
    assert all(len(r['plan']['commands']) == 30 and len(r['plan']['contexts']) == 6 for r in bundle['rows'])
