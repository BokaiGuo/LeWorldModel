"""RAM arithmetic mocks + real eight saved J preparation. Never execute CUDA."""
from contextlib import contextmanager
from copy import deepcopy
import json
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import p1_line_search_gn_fd_cuda_20261027_v1 as m


def context(z, h):
    return dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192, action_dim=10,
        latent_history=z.tolist(), action_history=h.tolist())


# Analytic affine recurrence in normalized RAW controls, plus tanh at the input.
# First-packet influence reaches step2 through BOTH latent (.7) and history (.3).
A = np.sin(np.arange(1920).reshape(192, 10)/37)/10
B = np.cos(np.arange(1920).reshape(192, 10)/41)/10


def toy_path(task, raw):
    z = np.asarray(task['root_context']['latent_history'], np.float32).astype(float)
    h = np.asarray(task['root_context']['action_history'], np.float32)
    paths = []
    for k in (0, 1):
        token = (np.asarray(raw[5*k:5*k+5], np.float32).reshape(10)-np.tile(np.asarray(task['action_mean'], np.float32), 5))/np.tile(np.asarray(task['action_std'], np.float32), 5)
        last = A@token if k == 0 else .7*z[-1]+.3*(A@h[-1])+B@token
        z = np.vstack((z[1:], last.astype(np.float32).astype(float)))
        h = np.vstack((h[1:], token))
        paths.append(context(z, h))
    return paths


def mock_protocol():
    tasks, matrices, points = {}, {}, []
    for r in (0, 1):
        task = dict(root_context=context(np.zeros((3, 192)), np.zeros((2, 10))), action_mean=[.1, -.2], action_std=[.7, 1.3])
        tasks[r] = task
        for it in range(1, 5):
            x = np.linspace(-.6, .5, 20).reshape(2, 10)*it/4
            x = x.astype(np.float32).astype(float)
            raw = np.tanh(x.astype(np.float32)).reshape(10, 2)
            j = np.column_stack((A, B))/np.tile(task['action_std'], 10)[None, :]*(1-raw.astype(float).reshape(20)**2)[None, :]
            path = toy_path(task, raw)
            point = dict(point_order=len(points), root_index=r, root_sha256=str(r)*64, task_id='mock', iteration=it,
                at_logits=x.tolist(), at_terminal=path[-1]['latent_history'][-1], saved_context_sha256=[m.screen.core.digest(c) for c in path],
                saved_Jv={d['id']:(j@np.asarray(d['vector'])).tolist() for d in m.directions()})
            matrices[len(points)] = j; points.append(point)
    return dict(points=points, directions=m.directions()), tasks, matrices


class ToyBackend:
    def __init__(self): self.active = False; self.guards = 0
    @contextmanager
    def guard(self, meter):
        assert not self.active
        self.active = True; self.guards += 1
        try: yield
        finally: self.active = False
    def rollout(self, task, raw, meter, mode):
        assert self.active and mode == 'whole_replay'
        paths = toy_path(task, raw)
        for k, c in enumerate(paths):
            for resource in ('action_encoder', 'predict'):
                token = meter.begin(resource); meter.end(token)
            meter.emit('native_packet', mode=mode, packet=k, context=c)
        return paths


def test_precommitted_directions_budget_and_zero_new_AD():
    ds = m.directions()
    assert [x['id'] for x in ds] == ['packet0_e0', 'packet1_e0', 'fixed20_sign_unit']
    assert all(abs(np.linalg.norm(x['vector'])-1) < 1e-14 for x in ds)
    assert ds[0]['vector'][0] == ds[1]['vector'][10] == 1
    assert all(v != 0 for v in ds[2]['vector'])
    assert m.EPS == (1e-3, 3e-3, 1e-2)
    b = m.budget()
    assert b['central_comparisons'] == 2*4*3*3 == 72
    assert b['predict'] == b['action_encoder'] == (8+2*72)*2 == 304
    assert b['line_search_plus_FD_predict'] == 460
    assert b['inherited_AD_forward_predicts_already_inside156'] == 16
    assert b['AD_calls'] == b['new_jacobians'] == 0


@pytest.mark.parametrize('eps', m.EPS)
@pytest.mark.parametrize('sign', m.SIGNS)
def test_round_once_and_tanh_once_without_ball_projection(eps, sign):
    p = dict(at_logits=np.full((2, 10), .5, np.float32).astype(float).tolist())
    d = m.directions()[0]['vector']
    out = m.perturb(p, d, eps, sign)
    x = (np.asarray(p['at_logits'])+eps*sign*np.asarray(d).reshape(2, 10)).astype(np.float32)
    assert np.array_equal(out['logits'], x)
    assert np.array_equal(out['commands'], np.tanh(x).reshape(10, 2))


def test_full_mock_workflow_counts_all_forward_and_restores_root_guards():
    p, tasks, js = mock_protocol(); log = []; meter = m.Meter(log.append)
    backends = {i:ToyBackend() for i in tasks}
    rows = m.evaluate(p, tasks, lambda i:backends[i], meter, log.append)
    report = m.analyze(p, rows, js)
    assert len(rows) == 152 and len(report['comparisons']) == 72
    assert len(report['epsilon_stability']) == 24
    assert report['nominal_anchors'] == 8
    assert meter.counts() == {k:dict(attempted=304, completed=304) for k in ('predict', 'action_encoder')}
    assert all(not b.active and b.guards == 1 for b in backends.values())
    assert report['scientific_pass_threshold'] is None and report['P1_admission'] is False
    # Mock arithmetic accuracy only; this is NOT a CUDA acceptance threshold.
    assert max(c['requested_direction_metrics']['relative_l2'] for c in report['comparisons']) < .003
    raw = m.canonical(report)
    assert len(raw) < 1024**2


@pytest.mark.parametrize('bug', ('missing_tanh_chain', 'double_tanh_chain', 'missing_normalizer', 'detached_history', 'detached_latent'))
def test_FD_distinguishes_shared_raw_row_self_consistency_from_wrong_J(bug):
    p, tasks, matrices = mock_protocol(); point = p['points'][3]
    j = matrices[3]; d = m.directions()[2]; eps = m.EPS[-1]
    sides = []
    for sign in m.SIGNS:
        v = m.perturb(point, d['vector'], eps, sign)
        v['terminal'] = toy_path(tasks[0], np.asarray(v['commands']))[-1]['latent_history'][-1]
        sides.append(v)
    chain = 1-np.tanh(np.asarray(point['at_logits'], np.float32)).astype(float).reshape(20)**2
    bad = j.copy()
    if bug == 'missing_tanh_chain': bad /= chain
    elif bug == 'double_tanh_chain': bad *= chain
    elif bug == 'missing_normalizer': bad *= np.tile(tasks[0]['action_std'], 10)
    elif bug == 'detached_history': bad[:, :10] *= .7
    else: bad[:, :10] *= .3
    bad_point = deepcopy(point); bad_point['saved_Jv'][d['id']] = (bad@np.asarray(d['vector'])).tolist()
    correct = m.compare(point, j, d, eps, *sides)
    incorrect = m.compare(bad_point, bad, d, eps, *sides)
    assert incorrect['requested_direction_metrics']['absolute_l2'] > 20*correct['requested_direction_metrics']['absolute_l2']


def test_zero_reference_has_null_ratios_no_fitted_floor():
    e = m.errors(np.zeros(192), np.zeros(192))
    assert e['relative_l2'] is e['symmetric_relative_l2'] is e['cosine'] is None


def test_effective_direction_accounts_for_float32_grid():
    p, tasks, js = mock_protocol(); point = p['points'][2]; d = m.directions()[2]
    sides = []
    for sign in m.SIGNS:
        v = m.perturb(point, d['vector'], m.EPS[0], sign)
        v['terminal'] = toy_path(tasks[0], v['commands'])[-1]['latent_history'][-1]; sides.append(v)
    r = m.compare(point, js[2], d, m.EPS[0], *sides)
    assert r['effective_direction_error_l2'] > 0
    assert np.array_equal(r['effective_float32_Jv'], js[2]@np.asarray(r['effective_direction']))


@pytest.mark.parametrize('damage', ('drop', 'swap', 'base', 'nominal'))
def test_all152_requests_and_eight_anchors_required(damage):
    p, tasks, js = mock_protocol(); meter = m.Meter(lambda e:None)
    backends = {i:ToyBackend() for i in tasks}
    rows = m.evaluate(p, tasks, lambda i:backends[i], meter, lambda e:None)
    if damage == 'drop': rows.pop()
    elif damage == 'swap': rows[1], rows[2] = rows[2], rows[1]
    elif damage == 'base': rows[1]['logits'][0][0] += .1
    else: rows[0]['terminal'][0] += .1
    with pytest.raises(ValueError): m.analyze(p, rows, js)


def test_history_contract_rejects_packet_token_substitution():
    p, tasks, js = mock_protocol(); task = tasks[0]
    raw = np.zeros((10, 2), np.float32); path = toy_path(task, raw)
    m.path_check(task, raw, path)
    bad = deepcopy(path); bad[1]['action_history'][0][0] += 1
    with pytest.raises(ValueError, match='history'): m.path_check(task, raw, bad)


def test_failure_nonrefund_no_duplicate_retry_or_false_sync_completion():
    p, _, _ = mock_protocol(); request = m.requests(p)[0]; log = []
    def sync(): raise RuntimeError('mock asynchronous CUDA failure')
    meter = m.Meter(log.append, sync=sync); meter.start(request)
    token = meter.begin('action_encoder')
    with pytest.raises(RuntimeError): meter.end(token)
    assert meter.counts()['action_encoder'] == dict(attempted=1, completed=0)
    with pytest.raises(ValueError): meter.start(request)
    with pytest.raises(ValueError): meter.begin('predict')


def test_grad_enabled_dispatch_rejected_before_charging():
    p, _, _ = mock_protocol(); meter = m.Meter(lambda e:None, no_grad=lambda:False)
    meter.start(m.requests(p)[0])
    with pytest.raises(ValueError, match='no-grad'): meter.begin('action_encoder')
    assert not meter.events


def test_failure_unwinds_native_root_guard_and_preserves_completed_calls():
    class WrongNominal(ToyBackend):
        def rollout(self, *args):
            paths = super().rollout(*args)
            paths[-1]['latent_history'][-1][0] += 1
            return paths
    p, tasks, _ = mock_protocol(); backend = WrongNominal(); meter = m.Meter(lambda e:None)
    with pytest.raises(ValueError, match='packet evidence'):
        m.evaluate(p, tasks, lambda i:backend, meter, lambda e:None)
    assert backend.active is False
    assert meter.counts() == {k:dict(attempted=2, completed=2) for k in ('predict', 'action_encoder')}


def ram_store():
    store = m.Store.__new__(m.Store)
    store.broken = store.closed = False
    store.comp = m.zlib.compressobj(6, m.zlib.DEFLATED, 31)
    chunks = []
    store.write = lambda name, raw, **kwargs:chunks.append(raw)
    return store, chunks


def test_gzip_partial_never_complete_before_footer_and_no_reopen():
    store, chunks = ram_store(); store.event(dict(kind='mock'))
    decoder = m.zlib.decompressobj(31)
    assert decoder.decompress(b''.join(chunks)) == m.canonical(dict(kind='mock'))
    assert decoder.eof is False
    store.close()
    decoder = m.zlib.decompressobj(31); decoder.decompress(b''.join(chunks))
    assert decoder.eof is True
    with pytest.raises(ValueError): store.close()
    with pytest.raises(ValueError): store.event(dict(kind='retry'))


def test_failed_journal_write_cannot_gain_false_complete_footer():
    store, _ = ram_store()
    def broken(*args, **kwargs): raise OSError('mock full disk')
    store.write = broken
    with pytest.raises(OSError): store.event(dict(kind='partial'))
    assert store.broken
    with pytest.raises(ValueError): store.close()


def test_shared_output_quota_and200MiB_floor_exact_boundaries():
    m.check_space(m.CAP-m.FINAL-1, 1, m.FLOOR+1)
    with pytest.raises(ValueError, match='2MiB'): m.check_space(m.CAP-m.FINAL, 1, m.FLOOR+1)
    m.check_space(m.CAP-1, 1, m.FLOOR+1, final=True)
    with pytest.raises(ValueError, match='2MiB'): m.check_space(m.CAP, 1, m.FLOOR+1, final=True)
    with pytest.raises(ValueError, match='200MiB'): m.check_space(0, 1, m.FLOOR)


def test_mock_end_to_end_compressed_journal_analysis_fit_shared2MiB():
    p, tasks, js = mock_protocol(); log = []; meter = m.Meter(log.append)
    backends = {i:ToyBackend() for i in tasks}
    rows = m.evaluate(p, tasks, lambda i:backends[i], meter, log.append)
    event_bytes = b''.join(m.canonical(e) for e in log)
    comp = m.zlib.compressobj(6, m.zlib.DEFLATED, 31)
    # Include per-event sync-flush overhead, not one unrealistically efficient gzip.
    compressed = b''.join(comp.compress(m.canonical(e))+comp.flush(m.zlib.Z_SYNC_FLUSH) for e in log)+comp.flush(m.zlib.Z_FINISH)
    analysis = m.gzip.compress(m.canonical(m.analyze(p, rows, js)), mtime=0)
    assert len(compressed)+len(analysis)+m.FINAL < m.CAP
    decoded = m.zlib.decompress(compressed, 31)
    assert decoded == event_bytes


def test_actual_saved8J_material_is_hash_bound_not_a_model_run():
    p, tasks, matrices = m.collect()
    assert len(p['points']) == len(matrices) == 8 and len(tasks) == 2
    assert [(x['root_index'], x['iteration']) for x in p['points']] == [(r, i) for r in (0, 1) for i in (1, 2, 3, 4)]
    assert len(m.requests(p)) == 152
    assert len(m.canonical(p)) < m.PREPARE_CAP
    assert p['scientific_pass_threshold'] is None and p['P1_admission'] is False
    assert p['inherited_line_search_resources']['predict'] == dict(attempted=156, completed=156)
    assert not any(name in sys.modules for name in ('torch', 'gymnasium', 'pymunk'))
