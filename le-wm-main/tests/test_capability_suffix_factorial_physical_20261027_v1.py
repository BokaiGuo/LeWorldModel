"""RAM-only arrays, journal and mocked env. No prepare/physics/neural execution."""
from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import time
import numpy as np
import pytest
from experiments.spectral_reachability import capability_suffix_factorial_physical_20261027_v1 as m
from test_moving_overlay_capability_probe_physical_20261027_v1 import fixture as probe_fixture, MemoryStore as ProbeStore
from test_p1_moving_overlay_recompile_physical_20261027_v1 import Env, fake_place, h


class MemoryStore(ProbeStore):
    def check(self, size=4096, final=False):
        m.require(self.used+size <= m.CAP-(0 if final else m.FINAL) and self.free >= m.FLOOR+size, 'RAM cap/floor')


@lru_cache(maxsize=1)
def frozen_fixture():
    return probe_fixture()


def producer_layout():
    original, data, images = deepcopy(frozen_fixture()); cases = []; rows = []; plans = []; forwards = []
    # Explicit RAM stand-in for a CUDA raw buffer. Adjacent float32 demonstrates
    # why physical code must not rederive raw from NumPy tanh(params).
    for anchor, plan in enumerate(original['plans']):
        for variant in m.VARIANTS:
            i = len(cases); scale = .1 if variant == 'teacher_small' else 1.
            value = np.nextafter(np.float32(.02 if scale == .1 else .2), np.float32(1.))
            suffix = np.full((5, 2), value, np.float32).tolist()
            case = dict(index=i, anchor=anchor, variant=variant, scale=scale,
                context=dict(latent_history=np.zeros((3, 192), np.float32).tolist(), action_history=np.zeros((2, 10), np.float32).tolist()),
                goal=np.ones(192, np.float32).tolist(), prefix=deepcopy(plan['commands'][:5]), original_plan=deepcopy(plan),
                **{k: plan[k] for k in ('root_slot', 'condition', 'arm', 'task_id', 'selection_digest')})
            winner = dict(index=3, raw=suffix, raw_receipt=dict(file=f'case{i:02}_actual_raw.npy.gz', file_sha256=h(('raw', i)),
                shape=[5, 2], dtype='float32', array_sha256=hashlib.sha256(np.asarray(suffix, np.float32).tobytes()).hexdigest()),
                params=[0.]*10, Q=.25, predicted=[.5]*192)
            replay_receipt = dict(winner['raw_receipt'], file=f'case{i:02}_replay_raw.npy.gz')
            for call in range(13):
                forwards.append(dict(index=len(forwards), case=i, completed=True, params=[0.]*10, raw=deepcopy(suffix),
                    raw_receipt=deepcopy(replay_receipt if call == 12 else winner['raw_receipt'] if call == 5 else dict(winner['raw_receipt'], file=f'raw{i}_{call}.npy.gz'))))
            rows.append(dict(index=i, status='sealed', result=dict(winner=winner, replay_exact=True, replay_raw_receipt=replay_receipt)))
            cases.append(case)
            plans.append(dict(index=i, anchor=anchor, variant=variant, scale=scale,
                **{k: plan[k] for k in ('root_slot', 'condition', 'arm', 'task_id', 'selection_digest')},
                original_plan=deepcopy(plan), prefix=deepcopy(case['prefix']), suffix=deepcopy(suffix),
                commands=case['prefix']+suffix, raw_receipt=deepcopy(winner['raw_receipt']), predicted_Q=.25, winner_index=3))
    contract = dict(anchors=8, cases=24, variants=list(m.VARIANTS))
    compiler = dict(schema=m.PRODUCER+'.report', status='all24_plans_sealed', errors=[], commands_released=False,
        P1_admission=False, contract=contract, rows=rows, forwards=forwards, predict_API=312,
        prepared='/RAM/compiler_prepare.json', prepare_sha256=h('compiler prepare'), artifact_sha256={})
    prep = dict(schema=m.PRODUCER+'.prepare', contract=contract, cases=cases, bindings={})
    verified = dict(schema=m.CONSUMER+'.report', status='all24_saved_verified', source_report='/RAM/compiler/report.json',
        source_report_sha256=h('compiler report'), prepare_sha256=compiler['prepare_sha256'], branches=24, known_roots=2,
        fresh_roots=0, plans=plans, commands_released=True, P1_admission=False, new_neural_calls=0, new_physics_calls=0)
    return compiler, prep, verified, original, data, images


def fixture():
    compiler, prep, verified, original, data, images = producer_layout()
    selected = m.validate_verified(compiler, prep, verified, original['plans'], h('compiler report'))
    p = dict(schema=m.NAME+'.prepare', contract=m.contract(), plans=original['plans'], anchors=original['anchors'],
        suffix_plans=selected, MAIN_verified24=True, bindings={}, compiler_report_sha256=h('compiler report'),
        consumer_report_sha256=h('MAIN accepted'), physical_report_sha256=h('original8 physical'),
        planner_report_sha256=original['planner_report_sha256'])
    return p, data, images


def execute(monkeypatch, *, p=None, factory=Env, started=None, store=None):
    fp, data, images = fixture(); p = fp if p is None else p
    store = MemoryStore() if store is None else store; store.free = m.FLOOR+m.CAP; envs = []
    def make(): e = factory(); envs.append(e); return e
    monkeypatch.setattr(m.base, 'place', fake_place)
    monkeypatch.setattr(m.base, 'validate_env', lambda _: None)
    monkeypatch.setattr(m.base, 'old_image', lambda ref: images[ref['path']].copy())
    monkeypatch.setattr(m, 'sha', lambda p: hashlib.sha256(store.files[Path(p).name]).hexdigest())
    def forbidden(*a, **k): raise AssertionError('old execute/probe/witness forbidden')
    monkeypatch.setattr(m.probe, 'execute', forbidden); monkeypatch.setattr(m.base, 'execute', forbidden)
    monkeypatch.setattr(m.base, 'episode', forbidden); monkeypatch.setattr(m.base.moving, 'witness', forbidden)
    r = m.execute(p, data, store, make, authorization=m.AUTH, prepared='/RAM/new_physical_prepare.json',
                  prepare_sha256=h('RAM prepare'), started=started)
    return r, store, envs


def test_true_layout_raw_false_requires_MAIN_verified24_not_v1():
    c, p, v, original, _, _ = producer_layout()
    selected = m.validate_verified(c, p, v, original['plans'], h('compiler report'))
    assert len(selected) == 24 and c['commands_released'] is False and v['commands_released'] is True
    assert selected[0]['plan']['suffix'][0][0] != 0.  # params0 would tanh0=0; no rederivation
    c['schema'] = 'capability_suffix_factorial_cuda_MAIN_20261027_v1.report'
    with pytest.raises(ValueError, match='v1 NEVER'): m.validate_verified(c, p, v, original['plans'], h('compiler report'))


@pytest.mark.parametrize('mutation', ['v1consumer', 'unreleased', 'missing', 'reorder', 'bool', 'anchor', 'root', 'prefix', 'suffixULP',
    'wrongwinner', 'missingreceipt', 'oldplan', 'scale', 'reportSHA', 'prepareSHA', 'planreceipt', 'forwardraw', 'replayraw'])
def test_actual_producer_layout_mutations_failclosed(mutation):
    c, p, v, o, _, _ = producer_layout()
    if mutation == 'v1consumer': v['schema'] = 'consume_suffix_factorial_MAIN_20261027_v1.report'
    elif mutation == 'unreleased': v['commands_released'] = False
    elif mutation == 'missing': v['plans'].pop()
    elif mutation == 'reorder': v['plans'][0], v['plans'][1] = v['plans'][1], v['plans'][0]
    elif mutation == 'bool': v['plans'][0]['index'] = False
    elif mutation == 'anchor': p['cases'][0]['anchor'] = 1
    elif mutation == 'root': v['plans'][0]['root_slot'] = 1
    elif mutation == 'prefix': v['plans'][0]['prefix'][0][0] = 0.
    elif mutation == 'suffixULP': v['plans'][0]['suffix'][0][0] = float(np.nextafter(np.float32(v['plans'][0]['suffix'][0][0]), np.float32(1.)))
    elif mutation == 'wrongwinner': v['plans'][0]['winner_index'] = 4
    elif mutation == 'missingreceipt': del c['rows'][0]['result']['winner']['raw_receipt']
    elif mutation == 'planreceipt': v['plans'][0]['raw_receipt']['file_sha256'] = h('wrong')
    elif mutation == 'forwardraw': c['forwards'][5]['raw'][0][0] = 0.
    elif mutation == 'replayraw': c['forwards'][12]['raw'][0][0] = 0.
    elif mutation == 'oldplan': v['plans'][0]['original_plan']['task_id'] = h('other')
    elif mutation == 'scale': p['cases'][2]['scale'] = 1.
    elif mutation == 'reportSHA': v['source_report_sha256'] = h('wrong')
    else: v['prepare_sha256'] = h('wrong')
    with pytest.raises(ValueError): m.validate_verified(c, p, v, o['plans'], h('compiler report'))


def test_full24_all51trace_counts_raw_suffix_and_no_setter(monkeypatch):
    p, _, _ = fixture(); r, store, envs = execute(monkeypatch, p=p)
    assert r['status'] == 'factorial24_suffix_physical_recorded_not_admitted', r['errors']
    assert len(r['rows']) == len(envs) == r['fixed_denominator'] == r['full_suffixes'] == 24
    assert r['counts'] == {k: dict(attempted=v, completed=v) for k, v in m.contract()['caps'].items()}
    assert len([x for x in store.files if x.endswith('.png')]) == 1584
    assert len(m.raw(r)) < m.FINAL and store.used < m.CAP
    for i, (row, env) in enumerate(zip(r['rows'], envs, strict=True)):
        assert env.closed and row['start_verified'] and row['start_state'] == p['anchors'][i//3]['checkpoint']['state']
        assert row['warmup_returned'] == 10 and row['prefix_returned'] == row['suffix_returned'] == 5
        assert len(env.commands) == 20 and env.dispatch_flags == [False]*20
        assert np.asarray(env.commands[10:15], np.float32).tobytes() == m.raw5(p['suffix_plans'][i]['plan']['prefix']).tobytes()
        assert np.asarray(env.commands[15:], np.float32).tobytes() == m.raw5(p['suffix_plans'][i]['plan']['suffix']).tobytes()
        assert [cp['step'] for cp in row['checkpoints']] == [0, 5, 10]
        assert len(row['final_state']) == 7 and row['missing_checkpoints'] == []
        assert '_set_state' not in env.__dict__ and 'reset' not in env.__dict__ and env.goal_pose.tolist() == [300., 300., 0.]
    assert not r['P1_admission'] and not r['scientific_admission'] and r['new_neural_calls'] == 0
    assert len(m.validate_saved(r, store.events, p)) == 24


@pytest.mark.parametrize('field', ['snapshot', 'state', 'nonselected_on', 'goal_pose', 'file_sha256'])
def test_any_original51trace_mismatch_blocks_first_suffix_all24_retained(monkeypatch, field):
    p, _, _ = fixture(); a = p['anchors'][0]
    if field == 'snapshot': a['checkpoint']['snapshot']['bodies']['block']['angular_velocity'] += 1e-9
    elif field == 'state': a['checkpoint']['state'][0] += 1e-9
    elif field == 'nonselected_on': a['trace'][1]['receipt']['pixel_sha256'] = h('bad')
    elif field == 'goal_pose': a['trace'][0]['pre']['goal_pose'][0] += 1e-9
    else: a['checkpoint']['file_sha256'] = h('bad')
    r, store, envs = execute(monkeypatch, p=p)
    assert len(r['rows']) == 24 and r['status'] == 'partial_or_failed'
    assert all(len(e.commands) == 15 for e in envs[:3])
    assert all(row['suffix_attempted'] == 0 and row['responses'] == [] for row in r['rows'][:3])
    assert all(row['status'] == 'suffix_completed' for row in r['rows'][3:])
    assert 'b0_plan_05_0.png' in store.files


@pytest.mark.parametrize('fault', ['throw', 'substitute', 'truncate', 'terminal'])
def test_suffix_partial_no_padding_actual_returns_and_cleanup(monkeypatch, fault):
    r, _, envs = execute(monkeypatch, factory=lambda: Env(fault=fault, at=16))
    assert len(r['rows']) == 24 and all(e.closed and len(e.commands) == 16 for e in envs)
    assert all(row['suffix_attempted'] == 1 and row['suffix_returned'] == (0 if fault == 'throw' else 1) for row in r['rows'])
    assert all(row['missing_checkpoints'] == [10] for row in r['rows'])
    if fault in ('truncate', 'terminal'):
        assert r['status'] == 'factorial24_suffix_physical_recorded_not_admitted', r['errors']
        assert r['early_stops'] == 24 and all(len(row['responses']) == 1 for row in r['rows'])
    else: assert r['status'] == 'partial_or_failed' and r['engineering_failures'] == 24
    if fault == 'throw': assert r['counts']['public_step'] == dict(attempted=384, completed=360)


def test_expired_before_constructor_retains24_unknown_labels(monkeypatch):
    r, _, envs = execute(monkeypatch, started=time.monotonic()-901)
    assert not envs and len(r['rows']) == 24 and r['counts']['constructor']['attempted'] == 0
    assert all(row['start_state'] is row['final_state'] is row['native_success'] is None for row in r['rows'])


def test_constructor_partial_unknown_and_primary_close_errors_retained(monkeypatch):
    def fail(): raise RuntimeError('unknown partial constructor')
    r, _, envs = execute(monkeypatch, factory=fail)
    assert not envs and r['counts']['constructor'] == dict(attempted=24, completed=0)
    assert r['total_internal_physics_scope'] == 'unknown_constructor_work'
    assert all(row['constructor_internal_work'] == 'unknown' for row in r['rows'])


def test_no_authorization_before_I_O_or_factory(monkeypatch):
    monkeypatch.setattr(m, 'verify', lambda *a: pytest.fail('unauthorized read'))
    with pytest.raises(ValueError): m.run('unused', h('x'), '/unused', lambda: pytest.fail('factory'), authorization='wrong')


def test_meter_caps_intents_actual_entry_return_failures_and_floor():
    store = MemoryStore(); store.free = m.FLOOR+m.CAP; meter = m.Meter(store)
    with pytest.raises(RuntimeError): meter.call('public_step', lambda: (_ for _ in ()).throw(RuntimeError('RAM failed')))
    assert meter.counts['public_step'] == dict(attempted=1, completed=0)
    meter.counts['constructor']['attempted'] = 24
    with pytest.raises(ValueError): meter.call('constructor', lambda: pytest.fail('cap bypass'))
    store.free = m.FLOOR-1
    with pytest.raises(ValueError): meter.call('public_step', lambda: pytest.fail('floor bypass'))
    store.free = m.FLOOR+m.CAP
    def fail(*a, **k): raise OSError('RAM intent failed')
    store.emit = fail
    with pytest.raises(OSError): meter.call('public_step', lambda: pytest.fail('no dispatch'))
    assert meter.counts['public_step'] == dict(attempted=1, completed=0)


def arguments():
    return dict(compiler_output='/RAM/compiler', compiler_report_sha256=h('compiler report'), producer_source_sha256=h('producer'),
        consumer_source_sha256=h('consumer'), consumer_report='/RAM/accepted.json', consumer_report_sha256=h('accepted'),
        physical_output='/RAM/original8', physical_report_sha256=h('physical'))


@pytest.mark.parametrize('mutation', [None, 'consumerSHA', 'reportSHA', 'acceptanceSHA', 'notverified', 'acceptanceMismatch'])
def test_compiler_material_calls_ONLY_v2_consumer_after_actual_external_seals(monkeypatch, mutation):
    c, p, v, _, _, _ = producer_layout(); args = arguments(); calls = []
    paths = {str(m.ROOT/'experiments/spectral_reachability'/(m.PRODUCER+'.py')): h('producer'),
             str(m.ROOT/'experiments/spectral_reachability'/(m.CONSUMER+'.py')): h('consumer'),
             '/RAM/compiler/report.json': h('compiler report'), '/RAM/accepted.json': h('accepted'), '/RAM/compiler_prepare.json': h('compiler prepare')}
    saved = deepcopy(v)
    if mutation == 'consumerSHA': paths[str(m.ROOT/'experiments/spectral_reachability'/(m.CONSUMER+'.py'))] = h('bad')
    elif mutation == 'reportSHA': paths['/RAM/compiler/report.json'] = h('bad')
    elif mutation == 'acceptanceSHA': paths['/RAM/accepted.json'] = h('bad')
    elif mutation == 'notverified': saved['commands_released'] = False; saved['status'] = 'raw'
    elif mutation == 'acceptanceMismatch': saved['plans'][0]['suffix'][0][0] = 0.
    monkeypatch.setattr(m, 'sha', lambda path: paths[str(path)])
    monkeypatch.setattr(m, 'read', lambda path, *a: saved if str(path).endswith('accepted.json') else c if str(path).endswith('report.json') else p)
    def consume(output, expected): calls.append((output, expected)); return v
    module = NS(__file__=str(m.ROOT/'experiments/spectral_reachability'/(m.CONSUMER+'.py')), consume=consume)
    monkeypatch.setattr(m.importlib, 'import_module', lambda name: (m.require(name.endswith(m.CONSUMER), 'v2 ONLY') or module))
    if mutation:
        with pytest.raises(ValueError): m.compiler_material(args)
        if mutation in ('consumerSHA', 'reportSHA', 'acceptanceSHA', 'notverified'): assert not calls
    else:
        _, _, received, bindings = m.compiler_material(args)
        assert received == v and calls == [(Path('/RAM/compiler'), h('compiler report'))]
        assert '/RAM/accepted.json' in bindings


def test_no_missing_futureSHA_fallback_or_v1_import(monkeypatch):
    args = arguments(); args['consumer_source_sha256'] = None
    monkeypatch.setattr(m.importlib, 'import_module', lambda *a: pytest.fail('no unfinished import'))
    with pytest.raises(ValueError, match='literal SHA'): m.compiler_material(args)


def test_no_neural_simulator_import_and_static_constructor_only():
    assert not any(n in sys.modules for n in ('torch', 'gymnasium', 'stable_worldmodel', 'pymunk'))
    assert m.constructor_contract()[1] == m.base.moving.ENV_SHA
    assert m.contract()['caps'] == dict(constructor=24, close=24, reset=48, public_step=480, physics_step=4896,
        set_state=96, goal_setter=72, goal_render=48, goal_bind=24, explicit_render=1584)
