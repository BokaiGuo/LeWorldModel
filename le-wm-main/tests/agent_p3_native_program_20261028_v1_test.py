"""RAM arithmetic and saved-metadata fixtures ONLY; no model/GPU/physics."""
from copy import deepcopy
import builtins
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_native_program_20261028_v1 as m


def task(slot):
    c = dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192,
        action_dim=10, latent_history=np.zeros((3, 192), np.float32).tolist(), action_history=np.zeros((2, 10), np.float32).tolist())
    t = dict(root_context=c, goal=np.ones(192).tolist(), projection=np.eye(192).tolist(),
        action_mean=[.125, -.25], action_std=[2., 4.])
    t['binding'] = {k: m.digest([slot, k]) for k in m.core.BINDINGS}
    t['binding'].update(goal_sha256=m.digest(t['goal']), projection_sha256=m.digest(t['projection']), context_sha256=m.digest(c),
        normalizer_sha256=m.digest([t['action_mean'], t['action_std']]), action_domain_sha256=m.digest(m.core.action_domain()))
    return m.core.validate_task(t)


def response(t, h):
    return dict(binding_sha256=m.digest(t['binding']), parent_context_sha256=m.digest(t['root_context']), horizon=h,
        nominal=np.zeros(192, np.float32).tolist(), jacobian=np.eye(192, 10*h, dtype=np.float32).tolist(), producer_sha256='a'*64)


def fixture():
    tasks = [task(s) for s in (0, 1)]; rows = []; branches = []
    for s in (0, 1):
        rs = [response(tasks[s], h) for h in (1, 3)]
        rows += [dict(slot=s, horizon=r['horizon'], status='returned', response=r) for r in rs]
        for mode, h in (('fixed', 1), ('adaptive', 3)):
            decision = dict(candidates=4, horizon=h, program_type='progress', program_depth_limit=3)
            branches.append(dict(slot=s, mode=mode, queue=dict(depth=0, parent_context_sha256=m.digest(tasks[s]['root_context']),
                decision=decision, requests=[dict(candidate_index=k, prefix_actions=[], prefix_edges=[]) for k in range(4)])))
    p = dict(tasks=tasks); jobs = m.jobs_from(p, dict(rows=rows, decision_rows=branches))
    return dict(tasks=tasks, jobs=jobs, profile=m.profile(jobs)), dict(rows=rows, decision_rows=branches)


class Sink:
    def __init__(self): self.events = []
    def event(self, e): self.events.append(deepcopy(e))


class RamReplay:
    """Synthetic numpy arithmetic, not frozen-model or CUDA evidence."""
    def __init__(self, task, store, fail=False): self.task, self.store, self.fail = task, store, fail

    def replay(self, parent, actions, segmented, ledger):
        path = []; phase = 'segmented' if segmented else 'whole'; step = None
        def arrays(kind, operation_index=None, **a):
            self.store.event(dict(kind=kind, scope=deepcopy(ledger.scope), phase=phase, step=step, operation_index=operation_index,
                arrays={k: m.signal.blob(v) for k, v in a.items()}))
        def call(resource, fn, inputs, shape):
            index = len(ledger.events)
            def actual():
                arrays('io_input', operation_index=index, **{resource+':'+k: v for k, v in inputs.items()})
                out = fn(); arrays('io_return', operation_index=index, **{resource+':output': out}); return out
            return ledger.call(resource, actual, shape=shape)
        state = np.asarray(parent['latent_history'], np.float32)[None]; history = np.asarray(parent['action_history'], np.float32)[None]
        mean, std = (np.asarray(self.task[k], np.float32) for k in ('action_mean', 'action_std'))
        for step, raw in enumerate(np.asarray(actions, np.float32).reshape(-1, 5, 2)):
            token = ((raw-mean)/std).reshape(10); outgoing = np.concatenate((history, token[None, None]), axis=1)
            arrays('program_packet', latent=state, history=history, raw=raw, mean=mean, std=std, token=token, outgoing=outgoing)
            def ae():
                if self.fail: raise RuntimeError('synthetic paid API fault')
                return np.repeat(outgoing.mean(axis=2, keepdims=True), 192, axis=2).astype(np.float32)
            encoded = call('action_encoder', ae, {'0': outgoing}, [1, 3, 10])
            def predict():
                pred = call('predictor', lambda: state+encoded, {'0': state, '1': encoded}, [1, 3, 192])
                proj = call('pred_proj', lambda: pred.reshape(3, 192), {'0': pred.reshape(3, 192)}, [3, 192])
                return proj.reshape(1, 3, 192)
            result = call('predict', predict, {'latent': state, 'act_emb': encoded}, [1, 3, 192])
            state = np.concatenate((state[:, 1:], result[:, -1:]), axis=1); history = outgoing[:, 1:]
            arrays('program_boundary', latent=state, history=history)
            path.append(dict(parent, latent_history=state[0].tolist(), action_history=history[0].tolist()))
        return path


def complete():
    p, _ = fixture(); store = Sink(); ledger = m.core.Ledger(p['profile']['resource_caps'], sink=store.event)
    rows = m.census(p['jobs']); store.event(dict(kind='denominator', rows=deepcopy(rows)))
    winners = m.panel(p, lambda s: RamReplay(p['tasks'][s], store), ledger, store, rows)
    r = dict(rows=rows, winners=winners, resource_receipt=ledger.receipt())
    return p, r, store.events


def test_complete_ram_panel_and_saved_array_closure():
    p, r, events = complete(); m.validate_saved(p, r, events)
    assert len(r['rows']) == 16 and len(r['winners']) == 4
    actual = r['resource_receipt']['completed']
    assert actual['predict_API'] == actual['predict_samples'] == 64
    assert actual['predict_context_tokens'] == 192 and actual['replay_calls'] == 32
    assert actual['array_SVD_calls'] == actual['solve_calls'] == 4
    assert actual['jacobian_API'] == actual['VJP_directions'] == actual['grad_API'] == 0
    assert all(w['executed_extension_depth'] == 1 and w['requested_depth'] == 3 for w in r['winners'])
    for mode, count in (('fixed', 16), ('adaptive', 48)):
        assert sum(e['delta']['predict_API'] for e in r['resource_receipt']['events'] if e['scope']['mode'] == mode) == count


@pytest.mark.parametrize('mutation', ['action', 'terminal', 'cost', 'winner', 'missing_row', 'missing_io', 'raw', 'normalizer', 'ae', 'predictor', 'projection', 'boundary', 'resource', 'scope', 'source_J', 'reorder', 'extra_event'])
def test_saved_mutations_rejected(mutation):
    p, r, events = complete()
    if mutation == 'action': r['rows'][0]['actions'][0][0] = .5
    elif mutation == 'terminal': r['rows'][0]['whole'][-1]['latent_history'][-1][0] += 1
    elif mutation == 'cost': r['rows'][0]['cost'] += 1
    elif mutation == 'winner': r['winners'][0]['index'] = 99
    elif mutation == 'missing_row': r['rows'].pop()
    elif mutation == 'missing_io': events.pop(next(i for i, e in enumerate(events) if e['kind'] == 'io_return'))
    elif mutation == 'resource': r['resource_receipt']['completed']['predict_API'] -= 1
    elif mutation == 'scope': next(e for e in events if e['kind'] == 'io_return')['scope']['slot'] = 99
    elif mutation == 'source_J': p['jobs'][0]['response']['jacobian'][0][0] = 5.
    elif mutation == 'reorder': r['rows'][0], r['rows'][1] = r['rows'][1], r['rows'][0]
    elif mutation == 'extra_event': events.append(dict(kind='outcome'))
    else:
        kind, key = {'raw': ('program_packet', 'raw'), 'normalizer': ('program_packet', 'mean'),
            'ae': ('io_return', 'action_encoder:output'), 'predictor': ('io_return', 'predictor:output'),
            'projection': ('io_return', 'pred_proj:output'), 'boundary': ('program_boundary', 'latent')}[mutation]
        e = next(e for e in events if e['kind'] == kind and key in e['arrays'])
        a = m.signal.unblob(e['arrays'][key]).copy(); a.flat[0] += 1; e['arrays'][key] = m.signal.blob(a)
    with pytest.raises((ValueError, KeyError)): m.validate_saved(p, r, events)


@pytest.mark.parametrize('field', ['binding_sha256', 'parent_context_sha256'])
def test_cross_root_child_signal_rejected(field):
    t = task(0); r = response(t, 1); r[field] = 'b'*64
    ledger = m.core.Ledger(m.profile(fixture()[0]['jobs'])['resource_caps']); ledger.scope = dict(test=True)
    with pytest.raises(ValueError, match='same actual root'): m.generate(t, r, ledger)
    assert not ledger.events


def test_bank_is_fixed_raw_float32_and_degenerate_collisions_retained():
    t = task(0); r = response(t, 3); r['jacobian'] = np.zeros((192, 30), np.float32).tolist()
    ledger = m.core.Ledger(m.profile(fixture()[0]['jobs'])['resource_caps']); ledger.scope = dict(test=True)
    bank = m.generate(t, r, ledger)
    assert len(bank['actions']) == 4 and all(np.array_equal(a, np.zeros((15, 2))) for a in bank['actions'])
    assert ledger.completed['solve_calls'] == ledger.completed['array_SVD_calls'] == 1


def test_native_wrong_device_before_import(monkeypatch):
    old = builtins.__import__; attempts = []
    def guarded(name, *a, **kw):
        if name.split('.')[0] == 'torch': attempts.append(name); raise AssertionError('forbidden neural import')
        return old(name, *a, **kw)
    monkeypatch.setattr(builtins, '__import__', guarded)
    with pytest.raises(ValueError, match='HOST CUDA'): m.NativeProgram(None, SimpleNamespace(type='cpu'), task(0), Sink())
    with pytest.raises(ValueError, match='MAIN explicit'): m.run(None, None, None, 'not_authorized')
    assert not attempts


def test_cold_import_no_torch_physics():
    script = 'import sys; from experiments.spectral_reachability import agent_p3_native_program_20261028_v1; assert not any(n.split(".")[0] in ("torch","gymnasium","pymunk") for n in sys.modules)'
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=m.ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_engineering_failure_full_denominator_nonrefund():
    p, _ = fixture(); rows = m.census(p['jobs']); sink = Sink(); ledger = m.core.Ledger(p['profile']['resource_caps'], sink=sink.event)
    with pytest.raises(RuntimeError, match='synthetic paid'):
        m.panel(p, lambda s: RamReplay(p['tasks'][s], sink, fail=True), ledger, sink, rows)
    assert len(rows) == 16 and rows[0]['status'] == 'failed' and all(r['cost'] is None for r in rows)
    assert ledger.attempted['action_encoder_calls'] == 1 and ledger.completed['action_encoder_calls'] == 0
    assert ledger.attempted['solve_calls'] == ledger.completed['solve_calls'] == 1


def test_reconfiguration_failclosed_no_fake_actions():
    p, _ = fixture()
    for j in p['jobs']: j['program_type'] = 'reconfiguration'
    p['profile'] = m.profile(p['jobs']); sink = Sink(); rows = m.census(p['jobs'])
    ledger = m.core.Ledger(p['profile']['resource_caps'], sink=sink.event)
    assert m.panel(p, lambda s: pytest.fail('no backend dispatch'), ledger, sink, rows) == []
    assert len(rows) == 16 and all(r['actions'] is None and r['cost'] is None and r['status'] == 'blocked_unsupported_program_type' for r in rows)
    assert not ledger.events


def test_no_padding_and_exact_predispatch_cap():
    p, _ = fixture(); caps = p['profile']['resource_caps'].copy(); caps['solve_calls'] = 0
    ledger = m.core.Ledger(caps); ledger.scope = dict(test=True)
    with pytest.raises(m.core.BudgetExceeded): m.generate(p['tasks'][0], p['jobs'][0]['response'], ledger)
    assert ledger.attempted['array_SVD_calls'] == ledger.attempted['solve_calls'] == 0


def test_saved_consume_roundtrip(tmp_path, monkeypatch):
    p, r, events = complete(); output = tmp_path/'saved'; output.mkdir()
    p.update(sources={}, expected_model_hash='d'*64, historical_paid_resources={'jacobian_API': 4}, historical_acquisition={'unknown': None})
    prepared = str(tmp_path/'prepare.json'); prep_sha = 'a'*64
    hashes = dict(original_state_dict='d'*64, all_buffers='b'*64)
    events += [dict(kind='model_before', hashes=hashes), dict(kind='model_after', hashes=hashes), dict(kind='sources_final', sha256=m.digest({}))]
    tail = '0'*64; raw = []
    for i, payload in enumerate(events):
        e = dict(index=i, previous=tail, payload=payload); tail = m.digest(e); raw.append(m.canonical(dict(e, sha256=tail))+b'\n')
    # RAM test temporary saved fixture; never a research result.
    (output/'events.jsonl').write_bytes(b''.join(raw))
    (output/'binding.json').write_bytes(m.canonical(dict(prepared=prepared, prepare_sha256=prep_sha)))
    r.update(schema=m.NAME+'.report', status='root_extensions_saved_not_admitted', prepared=prepared, prepare_sha256=prep_sha,
        errors=[], sources_verified=True, source_manifest_sha256=m.digest({}), model_before=hashes, model_after=hashes,
        saved_verification=dict(completed_upstream_passes=2, upstream_array_SVD_calls=8,
            partial_failed_pass_work_unknown=False, outside_native_generation_ledger=True),
        host={'RAM_fixture_not_CUDA': True}, wall_seconds=1., CUDA_peak_bytes=[0, 0], journal_tail=tail,
        artifact_sha256={n: m.sha(output/n) for n in ('events.jsonl','binding.json')}, **m.FLAGS)
    (output/'report.json').write_bytes(m.canonical(r)); monkeypatch.setattr(m, 'verify', lambda path, expected: p)
    result = m.consume(output, m.sha(output/'report.json'))
    assert result['denominator'] == 16 and result['scientific_effect'] is None and result['complete_tree'] is False


def test_original_contract_and_flags_not_rewritten():
    assert m.rule()['damping'] == .002 and m.rule()['candidates'] == 4
    assert all(v is False for v in m.FLAGS.values())
    assert m.sha(m.signal.__file__) == '6ce7722eca40cdd5f8958ba9590d8dc3d76d38bd10895d5ca494c1db3b6d100b'


def test_prepare_verify_metadata_roundtrip_and_mutation(tmp_path, monkeypatch):
    p, _ = fixture(); path = tmp_path/'prepare.json'; path.write_bytes(m.canonical(p))
    monkeypatch.setattr(m, 'material', lambda report, expected: deepcopy(p))
    original = deepcopy(p); original.update(report='fixed_saved_report', report_sha256='a'*64)
    p.update(report='fixed_saved_report', report_sha256='a'*64); path.write_bytes(m.canonical(p))
    expected = m.sha(path); assert m.verify(path, expected) == original
    with pytest.raises(ValueError, match='external prepare'): m.verify(path, '0'*64)
    p['profile']['resource_caps']['predict_API'] += 1
    with pytest.raises(ValueError, match='exact source'): m.verify(path, expected)


@pytest.mark.parametrize('fault', ['missing_task', 'missing_decision', 'reorder', 'child', 'quota', 'horizon', 'root_identity'])
def test_jobs_deny_identity_and_denominator_mutations(fault):
    p, source = fixture()
    if fault == 'missing_task': p['tasks'].pop()
    elif fault == 'missing_decision': source['decision_rows'].pop()
    elif fault == 'reorder': source['decision_rows'][0], source['decision_rows'][1] = source['decision_rows'][1], source['decision_rows'][0]
    elif fault == 'child': source['decision_rows'][0]['queue']['depth'] = 1
    elif fault == 'quota': source['decision_rows'][0]['queue']['decision']['candidates'] = 3
    elif fault == 'horizon': source['decision_rows'][0]['queue']['decision']['horizon'] = 2
    else: source['rows'][0]['response']['parent_context_sha256'] = 'a'*64
    with pytest.raises(ValueError): m.jobs_from(p, source)
