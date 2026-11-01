"""Pure saved-array, small PNG and fake-interface tests; no simulator/neural."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p3_joint_horizon_goal_metric_physical_20261027_v1 as m


def source_fixture():
    records = []
    for i in range(4):
        p = dict(root_sha256=str(i)*64, root_context=dict(m.planner.META,
            latent_history=np.zeros((3, 192)).tolist(), action_history=np.zeros((2, 10)).tolist()),
            goal=np.zeros(192).tolist(), projection=np.eye(192).tolist())
        p['input_sha256'] = m.digest(p)
        new = np.zeros((192, 192)); new[0, 0] = 1
        qs = dict(old_Q=p['projection'], goal_patch_Q=new.tolist())
        records.append(dict(packet=p, seed_index=i, metrics={q: dict(projection=v,
            active_metric_id=m.planner.active_id(p, q, v, {})) for q, v in qs.items()}))
    body = dict(records=records, arm_order=[list(m.ARMS[i:]+m.ARMS[:i]) for i in range(4)],
        normalization=dict(mean=[0., 0.], std=[1., 1.]))
    exports = []
    for r in records:
        for arm in m.ARMS:
            q = arm.split('__')[0]
            h = 2 if arm.endswith('uniform') else 3
            raw = np.linspace(-.3, .4, 10*h).reshape(5*h, 2)
            context = dict(deepcopy(r['packet']['root_context']),
                action_history=raw.astype(np.float32).reshape(h, 10)[-2:].astype(float).tolist())
            winner = dict(completed=True, horizon=h, index=1, actions=raw.tolist(), context=context, cost=0.)
            e = dict(root_sha256=r['packet']['root_sha256'], seed_index=r['seed_index'], arm=arm,
                active_metric_id=r['metrics'][q]['active_metric_id'], source_input_sha256=r['packet']['input_sha256'],
                root_context=r['packet']['root_context'], goal=r['packet']['goal'], projection=r['metrics'][q]['projection'],
                winner=winner, report_sha256='fixture-report', protocol_id=m.digest(body),
                P1_admission=False, P3_admission=False, physical_ready=False)
            e['selection_sha256'] = m.digest(e)
            exports.append(e)
    return body, exports


@pytest.fixture(scope='module')
def sources():
    return source_fixture()


def fixture_plan(h=2, root=0, arm=m.ARMS[0]):
    label, schedule = arm.split('__')
    p = dict(schema=m.NAME+'.plan', root_index=root, root_sha256=str(root)*64, method=arm,
        metric_label=label, schedule=schedule, active_metric_id='active-'+arm,
        planned_horizon_tokens=h, planned_commands=5*h, corrected_search_horizon=3,
        commands=np.zeros((5*h, 2)).tolist(), goal=np.zeros(192).tolist(),
        predictor_context=dict(m.planner.META, latent_history=np.zeros((3, 192)).tolist(), action_history=np.zeros((2, 10)).tolist()),
        selected_predicted_context=dict(m.planner.META, latent_history=np.zeros((3, 192)).tolist(), action_history=np.zeros((2, 10)).tolist()),
        predicted_costs=dict(old_Q=0., goal_patch_Q=0.))
    p['selection_sha256'] = m.digest(p)
    return p


class Env:
    def __init__(self, missing=None, fail=None, renderfail=None, terminate=None):
        self.n = 0
        self.missing, self.fail, self.renderfail, self.terminate = missing, fail, renderfail, terminate
    def render(self):
        if self.n == self.renderfail: raise RuntimeError('fixture render')
        rgb = np.zeros((224, 224, 3), np.uint8)
        if self.n != self.missing: rgb[5:7, 5:7] = [65, 105, 225]
        return rgb
    def step(self, action):
        self.n += 1
        if self.n == self.fail: raise RuntimeError('fixture step')
        return None, 0., self.n == self.terminate, False, {}


def branch(tmp_path, h=2, env=None, reset=lambda: dict(context='same'), p=None):
    return m.physical_branch(plan=p or fixture_plan(h), output=tmp_path, reset=reset, env=env or Env(),
        state_of=lambda e: np.full(7, e.n, float),
        evaluate=lambda s: dict(official=dict(official_state_distance=float(s[0])), pose=dict(pose_cost=float(s[0]))))


def test_sixteen_stable_identity_join_budget_and_no_old_endpoint(sources):
    body, exports = sources
    a = m.plans_from_exports(body, exports, 'fixture-report')
    b = m.plans_from_exports(body, list(reversed(exports)), 'fixture-report')
    assert a == b and len(a) == 16
    assert sum(p['planned_commands'] for p in a) == 200
    assert m.contract()['selected_path_step_cap'] == 240
    assert all('terminal_rgb_path' not in p for p in a)
    for p in a: m.validate_plan(p)


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'goal', 'Q', 'actions', 'history', 'context', 'report', 'root', 'hash', 'H', 'active'])
def test_export_binding_rejects_tampering(sources, damage):
    body, exports = deepcopy(sources)
    e = exports[0]
    if damage == 'missing': exports.pop()
    elif damage == 'duplicate': exports[1] = deepcopy(e)
    elif damage == 'goal': e['goal'][0] += 1
    elif damage == 'Q': e['projection'][0][0] += 1
    elif damage == 'actions': e['winner']['actions'][0][0] += 2
    elif damage == 'history': e['winner']['context']['action_history'][0][0] += .01
    elif damage == 'context': e['root_context']['action_dim'] = 2
    elif damage == 'report': e['report_sha256'] = 'wrong'
    elif damage == 'root': e['root_sha256'] = 'wrong'
    elif damage == 'H': e['winner']['horizon'] = 3
    elif damage == 'active': e['active_metric_id'] = 'wrong'
    else: e['selection_sha256'] = 'wrong'
    if damage not in ('missing', 'duplicate', 'hash'):
        e['selection_sha256'] = m.digest({k: v for k, v in e.items() if k != 'selection_sha256'})
    with pytest.raises((ValueError, KeyError)): m.plans_from_exports(body, exports, 'fixture-report')


@pytest.mark.parametrize('h', [2, 3])
def test_all_commands_and_fresh_durable_endpoint(tmp_path, h):
    r = branch(tmp_path, h); m.verify_branch(fixture_plan(h), r)
    assert r['physical_env_step_attempts'] == r['physical_env_step_returns'] == 5*h
    assert r['full_horizon_completed'] and r['endpoint_scope'] == f'full_H{h}'
    assert len(list(tmp_path.glob('*.png'))) == 5*h+1


@pytest.mark.parametrize('kwargs,a,b,fresh', [({'missing':0},0,0,True), ({'missing':6},6,6,True),
    ({'fail':7},7,6,False), ({'renderfail':7},7,7,False), ({'terminate':7},7,7,True)])
def test_partial_attempts_returns_and_no_stale_encoding(tmp_path, kwargs, a, b, fresh):
    r = branch(tmp_path, 3, Env(**kwargs)); m.verify_branch(fixture_plan(3), r)
    assert (r['physical_env_step_attempts'], r['physical_env_step_returns']) == (a, b)
    assert r['endpoint_fresh'] is fresh and not r['full_horizon_completed']


def test_bad_plan_before_reset(tmp_path):
    p = fixture_plan(); p['commands'][0][0] = 2
    with pytest.raises(ValueError):
        branch(tmp_path, p=p, reset=lambda: pytest.fail('must not reset'))


def test_failed_reset_is_durable_and_not_fresh(tmp_path):
    def fail(): raise ValueError('reset identity failed')
    r = branch(tmp_path, reset=fail)
    assert r['status'] == 'branch_failed' and not r['endpoint_fresh']
    assert r['physical_env_step_attempts'] == 0 and len(list(tmp_path.glob('*.physical.json'))) == 1


def test_journal_or_endpoint_tamper_refused(tmp_path):
    r = branch(tmp_path)
    Path(r['terminal_rgb_path']).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='RGB'): m.verify_branch(fixture_plan(), r)


def pairs_fixture(hs=(2, 3), partial=False):
    plans, rows = [], []
    for i in range(4):
        for arm in m.ARMS:
            h = hs[arm.startswith('goal_patch_Q')]
            p = fixture_plan(h, i, arm); plans.append(p)
            row = {k: deepcopy(p[k]) for k in ('root_index', 'root_sha256', 'method', 'selection_sha256',
                'planned_commands', 'planned_horizon_tokens', 'metric_label', 'schedule', 'active_metric_id')}
            row.update(context_identity=dict(same='context'), full_horizon_completed=not partial, endpoint_fresh=True,
                physical_env_step_returns=5 if partial else 5*h, pose=dict(pose_cost=2 if arm.startswith('goal_patch_Q') else 1))
            rows.append(row)
    return plans, rows


@pytest.mark.parametrize('hs,partial,stratum', [((2,2),False,'full_H2'), ((3,3),False,'full_H3'),
    ((2,3),False,'full_mixed_H2_H3'), ((2,3),True,'partial_same_steps')])
def test_separate_Q_and_schedule_contrasts_no_mixed_delta(hs, partial, stratum):
    pairs = m.comparisons(*pairs_fixture(hs, partial))
    assert len(pairs) == 16
    for p in pairs:
        if p['contrast'].startswith('Q_effect'):
            assert p['classification'] == stratum
            assert (p['right_minus_left_pose'] is None) == (partial or hs[0] != hs[1])
        else:
            assert p['left'].split('__')[0] == p['right'].split('__')[0]


def test_blocked_rows_have_complete_census():
    ps, rs = pairs_fixture()
    rows = []; m.materialize_missing(ps, rows)
    assert len(rows) == 16 and all(r['status'] == 'blocked' for r in rows)
    assert all(p['classification'] == 'partial_or_unmatched' for p in m.comparisons(ps, rows))


def encoding_protocol(plan):
    new = np.zeros((192, 192)); new[0, 0] = 1
    return dict(plans=[plan], root_metrics={plan['root_sha256']: dict(goal=plan['goal'],
        metrics=dict(old_Q=dict(projection=np.eye(192).tolist()), goal_patch_Q=dict(projection=new.tolist())))})


def test_no_encoding_setup_before_physical_report(tmp_path):
    with pytest.raises(ValueError):
        m.optional_encodings({}, [], tmp_path, lambda event: pytest.fail('early CUDA'))


@pytest.mark.parametrize('kwargs,n', [({},1), ({'missing':6},1), ({'fail':7},0), ({'renderfail':7},0)])
def test_optional_fresh_encodings_two_Q_one_vector(tmp_path, kwargs, n):
    p = fixture_plan(); r = branch(tmp_path, env=Env(**kwargs))
    m.write(tmp_path/'physical_report.json', dict(rows=[r]))
    setup_calls = []
    def setup(event):
        setup_calls.append(1)
        return lambda path: np.ones(192)
    receipt = m.optional_encodings(encoding_protocol(p), [r], tmp_path, setup)[0]
    assert receipt['attempted_encode_samples'] == n
    if n:
        assert receipt['actual_costs']['old_Q'] == 1 and receipt['actual_costs']['goal_patch_Q'] == 1/192
        assert receipt['endpoint_scope'] == r['endpoint_scope']
    else:
        assert receipt['status'] == 'blocked_stale_endpoint' and not setup_calls


@pytest.mark.parametrize('stage', ['setup', 'encode', 'shape', 'RGB'])
def test_optional_encoding_failure_retained(tmp_path, stage):
    p = fixture_plan(); r = branch(tmp_path)
    m.write(tmp_path/'physical_report.json', dict(rows=[r]))
    if stage == 'RGB': Path(r['terminal_rgb_path']).write_bytes(b'broken')
    def setup(event):
        if stage == 'setup': raise RuntimeError('setup failed')
        def encode(path):
            if stage == 'encode': raise RuntimeError('encode failed')
            return [0, 0]
        return encode
    receipt = m.optional_encodings(encoding_protocol(p), [r], tmp_path, setup)[0]
    assert receipt['status'] in ('encoding_setup_failed', 'encoding_failed') and receipt['failure']
    assert len(list(tmp_path.glob('*.encoding.json'))) == 1


def test_disk_denial_before_dispatch_not_physical_step(tmp_path, monkeypatch):
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda path: SimpleNamespace(free=210*m.MIB))
    with pytest.raises(ValueError, match='200 MiB'): m.disk_check(tmp_path)
    env = Env(); proxy = m.DiskGuardEnv(env, tmp_path)
    p = fixture_plan()
    r = m.physical_branch(plan=p, output=tmp_path, reset=lambda: {}, env=proxy,
        state_of=lambda e: np.zeros(7), evaluate=lambda s: {})
    assert env.n == 0 and r['physical_env_step_attempts'] == 0


def test_prepare_requires_actual_report_before_any_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'disk_check', lambda *a, **k: None)
    with pytest.raises(FileNotFoundError):
        m.prepare(tmp_path, tmp_path/'no-actual-planner', 'never-fake', tmp_path/'prepared')
    assert not (tmp_path/'prepared').exists()


def test_real_call_meter_accounts_collection_warmup_selected_separately():
    class Public:
        def __init__(self): self.n = self.r = 0
        def reset(self, *a, **kw): self.r += 1; return None, {}
        def step(self, action): self.n += 1; return None, 0, False, False, {}
    env = Public(); events = []
    meter = m.PhysicalMeter(env, events.append)
    for _ in range(16): meter.reset()
    with pytest.raises(ValueError): meter.step([0., 0.])
    for i in range(4):
        for arm in m.ARMS:
            p = fixture_plan(3, i, arm); meter.bind(p)
            meter.phase = 'warmup'; meter.reset()
            for _ in range(15): meter.step([0., 0.])
            with pytest.raises(ValueError): meter.step([0., 0.])
            meter.phase = 'selected'
            for _ in range(15): meter.step([0., 0.])
            with pytest.raises(ValueError): meter.step([0., 0.])
    r = meter.receipt()
    assert r['steps']['root_collection'] == dict(attempts=0, returns=0)
    assert r['steps']['warmup'] == r['steps']['selected'] == dict(attempts=240, returns=240)
    assert env.n == r['total_public_env_step_attempts'] == r['total_public_env_step_returns'] == 480
    assert env.r == r['total_env_reset_attempts'] == r['total_env_reset_returns'] == 32
    assert r['resets']['root_collection'] == r['resets']['warmup'] == dict(attempts=16, returns=16)
    assert len([e for e in events if e['kind'] == 'env_step' and e['completed']]) == 480


def test_failed_reset_and_warmup_call_attempts_not_refunded():
    class Failed:
        def reset(self): raise RuntimeError('reset failure')
        def step(self, action): raise RuntimeError('step failure')
    events = []; meter = m.PhysicalMeter(Failed(), events.append)
    with pytest.raises(RuntimeError): meter.reset()
    meter.bind(fixture_plan()); meter.phase = 'warmup'
    with pytest.raises(RuntimeError): meter.step([0., 0.])
    r = meter.receipt()
    assert r['resets']['root_collection'] == dict(attempts=1, returns=0)
    assert r['steps']['warmup'] == dict(attempts=1, returns=0)
    assert 'error' in events[-1] and events[-1]['completed'] is False


@pytest.mark.parametrize('kind,shape', [('encoder',(1,3,224,224)), ('projector',(1,192))])
def test_encoding_hook_failed_attempt_error_event_and_cap(kind, shape):
    stats = {kind+'_attempts':0, kind+'_completed':0}; log = []
    x = SimpleNamespace(shape=shape, device=SimpleNamespace(type='cuda'))
    def fail(x): raise RuntimeError('fixture callback failure')
    fn = m.counted_encoding(fail, kind, shape, log.append, stats)
    with pytest.raises(RuntimeError): fn(x)
    assert stats[kind+'_attempts'] == 1 and stats[kind+'_completed'] == 0
    assert len(log) == 2 and log[-1]['completed'] is False and 'error' in log[-1]
    fn = m.counted_encoding(lambda x: 'constant result', kind, shape, log.append, stats)
    for _ in range(15): fn(x)
    with pytest.raises(ValueError): fn(x)
    assert stats[kind+'_attempts'] == 16 and stats[kind+'_completed'] == 15


def test_encoding_primary_survives_failure_event_write_error():
    stats = dict(encoder_attempts=0, encoder_completed=0)
    def fail(x): raise RuntimeError('PRIMARY')
    def log(e):
        if 'error' in e: raise OSError('journal failed')
    fn = m.counted_encoding(fail, 'encoder', (1,3,224,224), log, stats)
    with pytest.raises(RuntimeError, match='PRIMARY'):
        fn(SimpleNamespace(shape=(1,3,224,224), device=SimpleNamespace(type='cuda')))
    assert stats['encoder_attempts'] == 1 and stats['encoder_completed'] == 0
    assert stats['journal_errors']
