"""Pure NumPy, constant interface fake and PNG tests. No neural/physics."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p2_composition_predict_frontier_physical_20261027_v1 as m
from experiments.spectral_reachability.d2p_composition_witness_v2 import PROVENANCE_KEYS


def source_fixture():
    cases=[]; source=[]; exports=[]
    norm=dict(mean=[0.,0.],std=[1.,1.])
    for i in range(4):
        key=str(i)*64
        initial=dict(m.planner.META,latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())
        root=dict(root_id=key,context=initial,root_rgb_sha256='a'*64,goal_rgb_sha256='b'*64)
        domain=dict(name='raw',units='fixture raw',normalization_sha256='c'*64,action_block=5,low=[-1.,-1.],high=[1.,1.])
        q=np.eye(192).tolist(); goal=np.zeros(192).tolist()
        compiled=dict(input_context=dict(emb=[initial['latent_history']],actions=[np.zeros((3,10)).tolist()]),
            goal_vectors=[goal],scoring=dict(metric=dict(projection=q,sha256='d'*64)))
        case=dict(root_index=i,root_sha256=key,root=root,action_domain=domain,
            goal_vector_sha256=m.frontier.digest(goal),projection_sha256=m.frontier.digest(q),metric_sha256='d'*64)
        cases.append(case);source.append(dict(root_sha256=key,compiled=compiled))
        for arm in m.ARMS:
            raw=np.linspace(-.3,.4,20).astype(np.float32).reshape(10,2)
            mid=deepcopy(initial);mid['action_history']=initial['action_history'][1:]+[raw[:5].reshape(10).tolist()]
            end=deepcopy(initial);end['action_history']=raw.reshape(2,10).tolist()
            contexts=[initial,end] if arm==m.ARMS[0] else [initial,mid,end]
            chunks=[raw.tolist()] if arm==m.ARMS[0] else [raw[:5].tolist(),raw[5:].tolist()]
            prov={k:'e'*64 for k in PROVENANCE_KEYS}
            witness=m.frontier.legacy().saved_witness(root,prov,domain,contexts,chunks,'fixture_67')
            exports.append(dict(root_index=i,root_sha256=key,arm=arm,horizon=2,raw_commands=raw.tolist(),
                root=deepcopy(root),action_domain=deepcopy(domain),provenance=prov,witness=witness,predicted_Q=0.,
                metric_sha256='d'*64,candidate_bank_sha256='f'*64,report_sha256='fixture-report',row_sha256='e'*64,resources={}))
    body=dict(context=dict(cases=cases,normalization=norm))
    exported=dict(schema=m.frontier.NAME+'.selected_actions',plans=exports,report_sha256='fixture-report',
        source_bound=True,saved_data_validation=True,P1_admission=False,P2_admission=False,total_resource_matched=False)
    return body,exported,dict(rows=source)


@pytest.fixture(scope='module')
def sources(): return source_fixture()


def fixture_plan(h=2,root=0,arm=m.ARMS[0]):
    p=dict(schema=m.NAME+'.plan',root_index=root,root_sha256=str(root)*64,method=arm,
        metric_label='source_Q',schedule='predict_frontier',active_metric_id='a'*64,
        planned_horizon_tokens=h,planned_commands=5*h,corrected_search_horizon=2,
        commands=np.zeros((5*h,2)).tolist(),goal=np.zeros(192).tolist(),
        predictor_context=dict(m.planner.META,latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist()),
        selected_predicted_context=dict(m.planner.META,latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist()),
        predicted_costs=dict(source_Q=0.))
    p['selection_sha256']=m.digest(p)
    return p


def test_all_eight_identity_join_and_no_old_endpoint(sources):
    body,ex,source=sources
    plans=m.plans_from_exports(body,ex,source,'fixture-report')
    reverse=deepcopy(ex);reverse['plans'].reverse()
    assert plans==m.plans_from_exports(body,reverse,source,'fixture-report')
    assert len(plans)==8 and sum(p['planned_commands'] for p in plans)==80
    assert all('terminal_rgb_path' not in p for p in plans)
    assert m.contract()['warmup_step_cap']==120 and m.contract()['total_public_env_step_cap']==200
    assert not m.contract()['P2_admission']
    for p in plans:m.validate_plan(p)


@pytest.mark.parametrize('damage',['old_schema','missing','duplicate','goal','Q','context','actions','history','metric','root','H','report','witness','source_bound'])
def test_source_and_new_export_contract_fail_closed(sources,damage):
    body,ex,source=deepcopy(sources); e=ex['plans'][0]
    if damage=='old_schema':ex['schema']='old40.selected_actions'
    elif damage=='missing':ex['plans'].pop()
    elif damage=='duplicate':ex['plans'][1]=deepcopy(e)
    elif damage=='goal':source['rows'][0]['compiled']['goal_vectors'][0][0]+=1
    elif damage=='Q':source['rows'][0]['compiled']['scoring']['metric']['projection'][0][0]=0
    elif damage=='context':e['root']['context']['latent_history'][0][0]=1
    elif damage=='actions':e['raw_commands'][0][0]=2
    elif damage=='history':e['witness']['receipt']['terminal_context']['action_history'][0][0]=.1
    elif damage=='metric':e['metric_sha256']='x'
    elif damage=='root':e['root_sha256']='x'
    elif damage=='H':e['horizon']=3
    elif damage=='report':e['report_sha256']='x'
    elif damage=='witness':e['witness']['receipt']['receipt_sha256']='x'
    else:ex['source_bound']=False
    with pytest.raises((ValueError,KeyError)):m.plans_from_exports(body,ex,source,'fixture-report')


def pairs_fixture(partial=False):
    plans=[];rows=[]
    for i in range(4):
        for arm in m.ARMS:
            p=fixture_plan(root=i,arm=arm);plans.append(p)
            r={k:deepcopy(p[k]) for k in ('root_index','root_sha256','method','selection_sha256','planned_commands',
                'planned_horizon_tokens','metric_label','schedule','active_metric_id')}
            r.update(context_identity=dict(same=True),full_horizon_completed=not partial,endpoint_fresh=True,failures=[],
                physical_env_step_returns=5 if partial else 10,
                pose=dict(pose_cost=2 if arm==m.ARMS[1] else 1),
                official=dict(official_state_distance=3 if arm==m.ARMS[1] else 1))
            rows.append(r)
    return plans,rows


@pytest.mark.parametrize('partial',[False,True])
def test_paired_identity_and_partial_no_full_delta(partial):
    pairs=m.comparisons(*pairs_fixture(partial))
    assert len(pairs)==4
    assert all(p['classification']==('partial_same_steps' if partial else 'full_H2') for p in pairs)
    assert all(p['right_minus_left_pose']==(None if partial else 1) for p in pairs)


def test_blocked_census():
    plans,_=pairs_fixture();rows=[];m.materialize_missing(plans,rows)
    assert len(rows)==8 and all(r['status']=='blocked' for r in rows)
    assert all(p['classification']=='partial_or_unmatched' for p in m.comparisons(plans,rows))


def test_foreign_horizon_refused_before_reset(tmp_path):
    with pytest.raises(ValueError):branch(tmp_path,p=fixture_plan(3),reset=lambda:pytest.fail('must not reset'))


@pytest.mark.parametrize('edge',[0,1])
def test_all_witness_program_ids_reconstructed_not_only_receipt(sources,edge):
    body,ex,source=deepcopy(sources)
    e=next(p for p in ex['plans'] if p['arm']==m.ARMS[1])
    e['witness']['edges'][edge]['program_id']='incorrect_edge_identity'
    with pytest.raises(ValueError,match='witness'):
        m.plans_from_exports(body,ex,source,'fixture-report')


def test_real_planning_material_saved_only_and_never_old40_exporter(monkeypatch):
    monkeypatch.setattr(m.frontier.legacy(),'selected_actions',lambda *a,**kw:pytest.fail('old40 exporter forbidden'))
    plans,refs,metrics,models,bindings,resources=m.planning_material(m.PROJECT,
        m.PROJECT/'results/p2_composition_predict_frontier_cuda_20261027_v1_main01',m.EXPECTED_REPORT_SHA)
    assert len(plans)==8 and sum(p['planned_commands'] for p in plans)==80
    assert resources['actual_unique_predict_calls']==1328
    assert resources['completed_operations']['jacobian_calls']==164
    assert len(metrics)==len(refs['references'])==4 and len(models)==2
    assert bindings[str(m.PROJECT/'results/p2_composition_predict_frontier_cuda_20261027_v1_main01/report.json')]==m.EXPECTED_REPORT_SHA
    assert {len(p['witness']['edges']) for p in plans}=={1,2}
    assert all('terminal_rgb_path' not in p for p in plans)


def test_foreign_external_report_sha_rejected_before_read(tmp_path):
    with pytest.raises(ValueError,match='externally provided'):
        m.planning_material(tmp_path,tmp_path/'absent','0'*64)


def encoding_protocol(plan):
    return dict(plans=[plan],root_metrics={plan['root_sha256']:dict(goal=plan['goal'],
        metrics=dict(source_Q=dict(projection=np.eye(192).tolist())))})

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



@pytest.mark.parametrize('h', [2])
def test_all_commands_and_fresh_durable_endpoint(tmp_path, h):
    r = branch(tmp_path, h); m.verify_branch(fixture_plan(h), r)
    assert r['physical_env_step_attempts'] == r['physical_env_step_returns'] == 5*h
    assert r['full_horizon_completed'] and r['endpoint_scope'] == f'full_H{h}'
    assert len(list(tmp_path.glob('*.png'))) == 5*h+1


@pytest.mark.parametrize('kwargs,a,b,fresh', [({'missing':0},0,0,True), ({'missing':6},6,6,True),
    ({'fail':7},7,6,False), ({'renderfail':7},7,7,False), ({'terminate':7},7,7,True)])
def test_partial_attempts_returns_and_no_stale_encoding(tmp_path, kwargs, a, b, fresh):
    r = branch(tmp_path, 2, Env(**kwargs)); m.verify_branch(fixture_plan(2), r)
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



def test_no_encoding_setup_before_physical_report(tmp_path):
    with pytest.raises(ValueError):
        m.optional_encodings({}, [], tmp_path, lambda event: pytest.fail('early CUDA'))


@pytest.mark.parametrize('kwargs,n', [({},1), ({'missing':6},1), ({'fail':7},0), ({'renderfail':7},0)])
def test_optional_fresh_encoding_original_Q_one_vector(tmp_path, kwargs, n):
    p = fixture_plan(); r = branch(tmp_path, env=Env(**kwargs))
    m.write(tmp_path/'physical_report.json', dict(rows=[r]))
    setup_calls = []
    def setup(event):
        setup_calls.append(1)
        return lambda path: np.ones(192)
    receipt = m.optional_encodings(encoding_protocol(p), [r], tmp_path, setup)[0]
    assert receipt['attempted_encode_samples'] == n
    if n:
        assert receipt['actual_costs']['source_Q'] == 1
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
        m.prepare(tmp_path, tmp_path/'no-actual-planner', m.EXPECTED_REPORT_SHA, tmp_path/'prepared')
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
            p = fixture_plan(2, i, arm); meter.bind(p)
            meter.phase = 'warmup'; meter.reset()
            for _ in range(15): meter.step([0., 0.])
            with pytest.raises(ValueError): meter.step([0., 0.])
            meter.phase = 'selected'
            for _ in range(10): meter.step([0., 0.])
            with pytest.raises(ValueError): meter.step([0., 0.])
    r = meter.receipt()
    assert r['steps']['root_collection'] == dict(attempts=0, returns=0)
    assert r['steps']['warmup'] == dict(attempts=120, returns=120)
    assert r['steps']['selected'] == dict(attempts=80, returns=80)
    assert env.n == r['total_public_env_step_attempts'] == r['total_public_env_step_returns'] == 200
    assert env.r == r['total_env_reset_attempts'] == r['total_env_reset_returns'] == 24
    assert r['resets']['root_collection'] == dict(attempts=16, returns=16)
    assert r['resets']['warmup'] == dict(attempts=8, returns=8)
    assert len([e for e in events if e['kind'] == 'env_step' and e['completed']]) == 200


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
    for _ in range(7): fn(x)
    with pytest.raises(ValueError): fn(x)
    assert stats[kind+'_attempts'] == 8 and stats[kind+'_completed'] == 7


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

