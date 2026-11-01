"""Full-sized NumPy engineering mocks, lossless RAM compression and tiny I/O faults.

No torch/model/simulator, no large pytest fixture files. Mock derivatives are not
evidence for the real frozen model. All dense rows and contexts are preserved.
"""
from copy import deepcopy
import gzip
import hashlib
import json
import sys
from types import SimpleNamespace
import zlib

import numpy as np
import pytest

from experiments.spectral_reachability import p1_line_search_gn_cuda_20261027_v1 as m


def task(seed=20261027):
    rng = np.random.default_rng(seed)
    return dict(environment='mock_only', task_contract_sha256='1'*64, model_sha256='2'*64,
        model_config_sha256='3'*64, metric_source_sha256='4'*64, goal_source_sha256='5'*64,
        representation='native_projected_cls', latent_dim=192, horizon=2, packet_steps=5, control_dim=2,
        action_mean=[.13, -.21], action_std=[.72, 1.13], packet_logit_cap=float(.75*np.sqrt(5)),
        feedback='open_loop_model_only', goal=rng.normal(size=192).tolist(), projection=np.eye(192).tolist(),
        root_raw_action_history=np.zeros((10, 2)).tolist(),
        root_context=dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192,
            action_dim=10, latent_history=rng.normal(size=(3, 192)).astype(np.float32).astype(float).tolist(),
            action_history=m.core.normalized_tokens(dict(action_mean=[.13, -.21], action_std=[.72, 1.13]),
                                                   np.zeros((10, 2))).astype(float).tolist()))


class Backend:
    """Dense analytic array stand-in, not a neural or authenticated CUDA backend."""
    def __init__(self, seed=7):
        self.a = np.random.default_rng(seed).normal(scale=.12, size=(192, 20))
        self.calls = []; self.jac_points = []

    def path(self, t, raw):
        before = deepcopy(t['root_context']); contexts = []; tokens = m.core.normalized_tokens(t, raw)
        for k in range(2):
            last = (np.asarray(before['latent_history'][-1])+self.a[:, k*10:(k+1)*10]@raw.reshape(20)[k*10:(k+1)*10]).astype(np.float32)
            after = dict(before, latent_history=before['latent_history'][1:]+[last.astype(float).tolist()],
                         action_history=[before['action_history'][-1], tokens[k].astype(float).tolist()])
            contexts.append(after); before = after
        return contexts

    def forward(self, t, raw, meter, kind, mode=None):
        path = self.path(t, raw)
        for k, c in enumerate(path):
            for resource in ('action_encoder', 'predict'):
                token = meter.begin(resource); meter.end(token)
            meter.emit(kind, packet=k, context=c, **({} if mode is None else dict(mode=mode)))
        return path

    def rollout(self, t, raw, meter, mode):
        self.calls.append((meter.group, mode))
        return self.forward(t, raw, meter, 'native_packet', mode)

    def jacobian(self, t, incumbent, meter):
        self.jac_points.append(deepcopy(incumbent))
        path = self.forward(t, np.asarray(incumbent['commands']), meter, 'jacobian_forward_packet')
        terminal = np.asarray(path[-1]['latent_history'][-1])
        meter.emit('jacobian_forward_complete', terminal=terminal.tolist(), logits=incumbent['logits'])
        rows = []
        for i in range(192):
            row = meter.call('vjp_directions', lambda: self.a[i].copy())
            rows.append(row); meter.emit('jacobian_row', row=i, raw_control_derivative=row.tolist())
        j = np.stack(rows)*(1-np.asarray(incumbent['commands']).reshape(20)**2)[None]
        meter.emit('joint_logit_jacobian', terminal=terminal.tolist(), jacobian=j.tolist())
        return terminal, j


class RAMStore(m.Store):
    """Same production compressor/put/event/close, memory-only _write transport."""
    def __init__(self):
        self.files = {}; self.comp = zlib.compressobj(6, zlib.DEFLATED, 31)
        self.broken = self.closed = False; self.raw_hash = hashlib.sha256(); self.raw_bytes = 0
        self.peak_nonfinal = 0; self._write('events.jsonl.gz', b'')

    def _write(self, name, raw, append=False, final=False):
        used = sum(map(len, self.files.values()))
        assert used+len(raw) <= m.CAP-(0 if final else m.FINAL), 'actual production quota exceeded in RAM'
        assert append or name not in self.files
        self.files[name] = self.files.get(name, b'')+raw
        if not final: self.peak_nonfinal = max(self.peak_nonfinal, used+len(raw))

    def event(self, e):
        raw = m.canonical(e); self.raw_hash.update(raw); self.raw_bytes += len(raw)
        super().event(e)


def full_mock():
    store = RAMStore(); rows = []; meters = []
    store.put('binding.json', dict(prepared='/mock/prepare.json', prepare_sha256='f'*64))
    for index in range(2):
        t = task(20261027+index); backend = Backend(7+index)
        def sink(e, index=index): store.event(dict(root_index=index, root_sha256=str(index)*64, **e))
        meter = m.Meter(sink); meters.append(meter)
        store.event(dict(kind='root_started', root_index=index, task_id=m.core.task_id(t)))
        row = dict(root_index=index, root_sha256=str(index)*64, **m.optimize(t, backend, meter))
        assert len(backend.calls) == 35  # initial3 +4*(5 candidates+3 replays)
        assert len(backend.jac_points) == 4
        store.put(f'root_{index}.json.gz', row); rows.append(row)
    store.close()
    resources = {k:dict(attempted=2*v, completed=2*v) for k,v in m.budget().items()}
    store.put('report.json', dict(schema=m.NAME+'.report', status='mock_only', resources=resources,
        prepare_sha256='f'*64, errors=[], first_error=None, model_hash_before='a'*64, model_hash_after='a'*64,
        model_unchanged=True, source_verified=True, host={'mock':True},
        artifact_sha256={k:hashlib.sha256(v).hexdigest() for k,v in store.files.items()},
        roots=[dict(root_sha256=r['root_sha256'], steps=r['steps'], final_cost=r['final_prediction']['cost']) for r in rows],
        inherited_resources_are_separate=True, new_images=0, new_physics=0, training=False, new_Q=False,
        independent_sample_increment=0, P1_admission=False, scientific_pass_threshold=None), final=True)
    return store, rows, meters


def test_full_two_root_capacity_and_lossless_readback():
    store, rows, meters = full_mock()
    raw = gzip.decompress(store.files['events.jsonl.gz'])
    assert hashlib.sha256(raw).hexdigest() == store.raw_hash.hexdigest()
    assert len(raw) == store.raw_bytes
    events, eof = m.recover(store.files['events.jsonl.gz']); assert eof
    assert len([e for e in events if e['kind'] == 'jacobian_row']) == 1536
    assert len([e for e in events if e['kind'] == 'joint_logit_jacobian']) == 8
    assert len([e for e in events if e['kind'] == 'derivative_complete']) == 8
    assert len([e for e in events if e['kind'] == 'native_packet']) == 140
    assert len([e for e in events if e['kind'] == 'jacobian_forward_packet']) == 16
    assert len([e for e in events if e['kind'] == 'evaluation_request']) == 70
    assert len([e for e in events if e['kind'] == 'line_proposals_frozen']) == 8
    for index in range(2):
        assert json.loads(gzip.decompress(store.files[f'root_{index}.json.gz'])) == rows[index]
        assert meters[index].counts() == {k:dict(attempted=v, completed=v) for k,v in m.budget().items()}
    for e in events:
        if e['kind'] == 'line_proposals_frozen': assert e['scales'] == list(m.SCALES)
    # Independent reconstruction from decoded rows, not only a count/check flag.
    for index in range(2):
        for iteration in range(1, 5):
            group = [e for e in events if e.get('root_index') == index and
                     e.get('evaluation_id') == f'iter_{iteration}_AD']
            row_events = [e for e in group if e['kind'] == 'jacobian_row']
            assert [e['row'] for e in row_events] == list(range(192))
            raw_j = np.asarray([e['raw_control_derivative'] for e in row_events])
            d = next(e['derivative'] for e in group if e['kind'] == 'derivative_complete')
            commands = np.tanh(np.asarray(d['at_logits'], np.float32)).reshape(20).astype(float)
            rebuilt = raw_j*(1-commands**2)[None]
            assert np.array_equal(rebuilt, d['raw_terminal_jacobian'])
            assert np.array_equal(rebuilt, next(e['jacobian'] for e in group if e['kind'] == 'joint_logit_jacobian'))
            proposal = next(e for e in group if e['kind'] == 'line_proposals_frozen')
            trials = m.line_trials(d['at_logits'], proposal['endpoint'], task()['packet_logit_cap'])
            assert np.array_equal(trials, proposal['logits'])
            selected = next(e for e in events if e.get('root_index') == index and e['kind'] == 'selection_complete' and
                            e['iteration'] == iteration)
            costs = [selected['incumbent_cost']]+selected['trial_costs']
            assert selected['selected_index'] == min(range(6), key=lambda k:(costs[k], k))
    size = sum(map(len, store.files.values()))
    assert size < m.CAP and store.peak_nonfinal < m.CAP-m.FINAL
    # Engineering capacity measurement, not a worst-case compression theorem.
    print('CAPACITY', json.dumps(dict(raw_journal_bytes=len(raw), total_output_bytes=size,
        journal_gzip_bytes=len(store.files['events.jsonl.gz']), cap=m.CAP, remaining_bytes=m.CAP-size)))


def test_interpolation_fixed_complete_feasible_and_exact_tanh():
    x = np.full((2, 10), .125); y = np.full((2, 10), -.25)
    trials = m.line_trials(x, y, 1.)
    assert len(trials) == 5 and np.array_equal(trials[0], y)
    for scale, trial in zip(m.SCALES, trials):
        assert np.array_equal(trial, (x+scale*(y-x)).astype(np.float32))
    with pytest.raises(ValueError): m.line_trials(np.ones((2, 10)), y, 1.)


def test_select_keeps_incumbent_tie_and_requires_every_trial():
    incumbent = dict(cost=1.)
    assert m.select(incumbent, [dict(cost=1.) for _ in range(5)]) == 0
    assert m.select(incumbent, [dict(cost=v) for v in (2., .5, .25, .1, .2)]) == 4
    with pytest.raises(ValueError): m.select(incumbent, [dict(cost=.1)])
    with pytest.raises(ValueError): m.select(incumbent, [dict(cost=float('nan'))]*5)


def test_no_early_exit_even_all_tie(monkeypatch):
    t = task(); backend = Backend(); events = []
    monkeypatch.setattr(m.core, 'model_cost', lambda t, z: 1.)
    meter = m.Meter(events.append); result = m.optimize(t, backend, meter)
    assert len(result['steps']) == 4 and all(s['selected_index'] == 0 for s in result['steps'])
    assert len(backend.calls) == 35 and len(backend.jac_points) == 4
    assert all(np.array_equal(p['logits'], np.zeros((2, 10))) for p in backend.jac_points)


def test_new_J_at_each_selected_point():
    t = task(); backend = Backend(); events = []; meter = m.Meter(events.append)
    m.optimize(t, backend, meter)
    updates = [e['prediction'] for e in events if e['kind'] == 'update_verified']
    for i in range(1, 4): assert backend.jac_points[i] == updates[i-1]
    assert any(np.any(np.asarray(p['logits'])) for p in backend.jac_points[1:])


def test_failed_attempt_no_refund_or_retry():
    meter = m.Meter(lambda e: None)
    meter.begin('action_encoder')
    assert meter.counts()['action_encoder'] == dict(attempted=1, completed=0)
    with pytest.raises(ValueError): meter.begin('predict')


def test_sync_error_does_not_mark_completed():
    def fail(): raise OSError('sync fault')
    meter = m.Meter(lambda e: None, fail); token = meter.begin('predict')
    with pytest.raises(OSError): meter.end(token)
    assert meter.counts()['predict'] == dict(attempted=1, completed=0)


def test_failed_vjp_retains_forward_and_partial_rows():
    events = []; meter = m.Meter(events.append); backend = Backend(); t = task()
    incumbent = m.verified(t, np.zeros((2, 10)), backend, meter)
    old = meter.call
    def broken(resource, fn):
        if resource == 'vjp_directions' and meter.counts()[resource]['attempted'] == 3:
            meter.begin(resource); raise RuntimeError('AD fault')
        return old(resource, fn)
    meter.call = broken
    with pytest.raises(RuntimeError): meter.call('jacobian', lambda: backend.jacobian(t, incumbent, meter))
    assert meter.counts()['predict'] == dict(attempted=8, completed=8)
    assert meter.counts()['jacobian'] == dict(attempted=1, completed=0)
    assert meter.counts()['vjp_directions'] == dict(attempted=4, completed=3)
    assert len([e for e in events if e['kind'] == 'jacobian_row']) == 3
    with pytest.raises(ValueError): meter.begin('predict')


def test_bad_context_or_replay_fail_closed():
    class Corrupt(Backend):
        def rollout(self, t, raw, meter, mode):
            path = super().rollout(t, raw, meter, mode)
            path[0]['action_history'][-1][0] += .1
            return path
    meter = m.Meter(lambda e: None)
    with pytest.raises(ValueError): m.verified(task(), np.zeros((2, 10)), Corrupt(), meter)
    assert meter.counts()['predict'] == dict(attempted=2, completed=2)


def test_whole_replay_mismatch_keeps_paid_work():
    class Corrupt(Backend):
        def rollout(self, t, raw, meter, mode):
            path = super().rollout(t, raw, meter, mode)
            if mode == 'whole_replay': path[-1]['latent_history'][-1][0] += .1
            return path
    meter = m.Meter(lambda e: None)
    with pytest.raises(ValueError, match='triple replay mismatch'):
        m.verified(task(), np.zeros((2, 10)), Corrupt(), meter)
    assert meter.counts()['predict'] == dict(attempted=6, completed=6)


def test_partial_gzip_and_footer_corruption():
    store = RAMStore()
    for i in range(3): store.event(dict(kind='row', i=i, dense=np.arange(20).tolist()))
    prefix = store.files['events.jsonl.gz']; events, eof = m.recover(prefix)
    assert len(events) == 3 and not eof
    store.close(); full = store.files['events.jsonl.gz']
    assert m.recover(full)[1] and not m.recover(full[:-5])[1]
    with pytest.raises(ValueError): m.recover(full+b'garbage')
    damaged = bytearray(full); damaged[-8] ^= 1
    with pytest.raises(zlib.error): m.recover(bytes(damaged))


def test_fsync_failure_preserves_first_error_and_marks_stream_broken(tmp_path, monkeypatch):
    store = m.Store(tmp_path/'tiny')
    store.event(dict(kind='before'))
    def fail(fd): raise OSError('fsync fault')
    monkeypatch.setattr(m.os, 'fsync', fail)
    with pytest.raises(OSError, match='fsync fault'): store.event(dict(kind='during'))
    assert store.broken
    events, eof = m.recover((store.output/'events.jsonl.gz').read_bytes())
    assert events[0] == dict(kind='before') and not eof
    # Failed fsync can leave readable bytes, but cannot claim durable success.
    used = (store.output/'events.jsonl.gz').stat().st_size
    with pytest.raises(ValueError): store.event(dict(kind='retry'))
    with pytest.raises(ValueError): store.close()
    assert (store.output/'events.jsonl.gz').stat().st_size == used


def test_quota_and_floor_before_writing(tmp_path, monkeypatch):
    store = m.Store(tmp_path/'tiny')
    with pytest.raises(ValueError): store._write('too_big', bytes(m.CAP))
    assert not (store.output/'too_big').exists()
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda p: type('D', (), {'free':m.FLOOR})())
    with pytest.raises(ValueError): store.put('floor.json', dict(a=1))
    assert not (store.output/'floor.json').exists()


def test_real_store_tiny_complete_readback_and_fsync(tmp_path, monkeypatch):
    syncs = []; original = m.os.fsync
    def fsync(fd): syncs.append(fd); original(fd)
    monkeypatch.setattr(m.os, 'fsync', fsync)
    store = m.Store(tmp_path/'tiny')
    event = dict(kind='dense', matrix=np.random.default_rng(9).normal(size=(3, 7)).tolist())
    store.event(event); store.put('root_0.json.gz', event); store.close()
    store.put('report.json', dict(status='mock_only'), final=True)
    assert len(syncs) == 5  # empty journal, event, object, footer, report
    assert m.recover((store.output/'events.jsonl.gz').read_bytes()) == ([event], True)
    assert json.loads(gzip.decompress((store.output/'root_0.json.gz').read_bytes())) == event


def test_failed_fsync_bytes_still_charged_against_quota(tmp_path, monkeypatch):
    store = m.Store(tmp_path/'tiny'); old = m.os.fsync
    def fail(fd): raise OSError('fsync fault')
    monkeypatch.setattr(m.os, 'fsync', fail)
    with pytest.raises(OSError): store._write('partial', b'paid')
    assert (store.output/'partial').stat().st_size == 4
    monkeypatch.setattr(m.os, 'fsync', old)
    monkeypatch.setattr(m, 'CAP', m.FINAL+6)
    with pytest.raises(ValueError): store._write('new', b'abc')
    assert not (store.output/'new').exists()


def test_run_pre_model_failure_keeps_first_error_and_unknown_model(tmp_path, monkeypatch):
    # No torch installation imported: explicit stub only, preflight always fails.
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'torch.nn', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'torch.nn.attention', SimpleNamespace(SDPBackend=None, sdpa_kernel=None))
    def no_load(*args, **kwargs): raise AssertionError('model load must not be reached')
    monkeypatch.setitem(sys.modules, 'experiments.checkpoint_compat', SimpleNamespace(load_compatible_pretrained=no_load))
    def preflight(*args): raise RuntimeError('first preflight fault')
    monkeypatch.setitem(sys.modules, 'experiments.gpu_preflight', SimpleNamespace(require_cuda_allocation=preflight))
    count = [0]
    def verify(*args):
        count[0] += 1
        if count[0] == 2: raise RuntimeError('secondary source fault')
        return dict(roots=[])
    monkeypatch.setattr(m, 'verify', verify)
    def close(self): raise RuntimeError('secondary footer fault')
    monkeypatch.setattr(m.Store, 'close', close)
    out = tmp_path/'failure'
    with pytest.raises(ValueError, match='partial line search preserved'):
        m.run(tmp_path/'prepare.json', 'e'*64, out)
    report = json.loads((out/'report.json').read_bytes())
    assert report['status'] == 'partial_or_failed'
    assert report['first_error']['message'] == 'first preflight fault'
    assert [e['stage'] for e in report['errors']] == ['execution', 'final_source', 'journal_finalize']
    assert report['model_unchanged'] is None and report['model_hash_before'] is None
    assert all(r == dict(attempted=0, completed=0) for r in report['resources'].values())
    assert not m.recover((out/'events.jsonl.gz').read_bytes())[1]


def test_saved_bindings_and_small_manifest():
    p = m.material()
    assert len(m.canonical(p)) < 512*1024
    assert p['total_budget']['predict'] == p['total_budget']['action_encoder'] == 156
    assert p['total_budget']['vjp_directions'] == 1536
    assert len(p['roots']) == 2 and p['scales'] == [1., .5, .25, .125, .0625]
    assert p['new_images'] == p['new_physics'] == 0 and p['P1_admission'] is False
    assert not any('stagnation' in path for path in p['bindings_sha256'])
