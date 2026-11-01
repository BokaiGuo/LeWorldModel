"""Pure NumPy/mock protocol and fault tests; NEVER neural/physics execution."""
from contextlib import contextmanager
from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from experiments.spectral_reachability import p1_two_stage_witness_20261027_v1 as m


def task():
    t=dict(environment='array_fixture_not_PushT',task_contract_sha256='f'*64,
        root_raw_action_history=np.zeros((10,2)).tolist(),goal=[.15]*10+[0.]*182,
        projection=np.eye(192).tolist(),model_sha256='a'*64,model_config_sha256='b'*64,
        metric_source_sha256='c'*64,goal_source_sha256='d'*64,representation='native_projected_cls',
        latent_dim=192,horizon=2,packet_steps=5,control_dim=2,action_mean=[0.,0.],action_std=[.2,.25],
        packet_logit_cap=float(.75*np.sqrt(5)),feedback='open_loop_model_only')
    t['root_context']=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,
        action_dim=10,latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())
    return t


class ArrayBackend:
    """State-dependent transition, analytic derivatives; not CPU neural code."""
    def __init__(self, fault=None):
        self.b=np.zeros((192,10));self.b[:10]=np.eye(10)
        self.fault=fault;self.calls=[];self.suffix_contexts=[];self.closed=False

    @contextmanager
    def guard(self,meter):
        try:yield
        finally:self.closed=True

    def packet(self,t,c,logits,meter,label):
        self.calls.append((meter.phase,meter.stage,label))
        raw=np.tanh(np.asarray(logits,np.float32)).reshape(5,2)
        token=m.core.normalized_tokens(t,raw)[0]
        meter.call('action_encoder',lambda:None)
        def predict():
            if self.fault=='prefix' and label=='actual_prefix':raise RuntimeError('first_prefix_error')
            z=np.asarray(c['latent_history'][-1]);gain=1+.05*z[0]
            return .7*z+gain*(self.b@token)
        z=meter.call('predict',predict)
        after=dict(c,latent_history=c['latent_history'][1:]+[z.tolist()],
            action_history=c['action_history'][1:]+[token.tolist()])
        m.edge(t,c,logits,after)
        meter.emit('mock_packet',label=label,context=after)
        return after

    def rollout(self,t,raw,meter,mode):
        # Recover exact logits only for the mock packet interface. Replacing
        # tanh(arctanh(raw)) could round differently; construct path directly.
        c=deepcopy(t['root_context']);paths=[]
        for k,token in enumerate(m.core.normalized_tokens(t,raw)):
            meter.call('action_encoder',lambda:None)
            z=np.asarray(c['latent_history'][-1])
            new=meter.call('predict',lambda:.7*z+(1+.05*z[0])*(self.b@token))
            c=dict(c,latent_history=c['latent_history'][1:]+[new.tolist()],
                action_history=c['action_history'][1:]+[token.tolist()])
            paths.append(deepcopy(c));self.calls.append((meter.phase,meter.stage,mode))
        return paths

    def jacobian(self,t,inc,meter):
        raw=np.asarray(inc['commands']);path=self.rollout(t,raw,meter,'root_AD')
        tokens=m.core.normalized_tokens(t,raw);z0=np.asarray(t['root_context']['latent_history'][-1])
        z1=np.asarray(path[0]['latent_history'][-1]);std=np.tile(np.asarray(t['action_std'],np.float32),5).astype(float)
        f=.7*np.eye(192);f[:,0]+=.05*(self.b@tokens[1])
        j=np.concatenate((f@((1+.05*z0[0])*self.b/std),(1+.05*z1[0])*self.b/std),axis=1)
        j*=1-raw.reshape(20)**2
        for _ in range(192):meter.call('vjp_directions',lambda:None)
        return path[-1]['latent_history'][-1],j

    def suffix_jacobian(self,t,c,nominal,meter):
        self.suffix_contexts.append(deepcopy(c))
        actual=self.packet(t,c,[0.]*10,meter,'local_AD')
        meter.emit('suffix_AD_forward',context=actual,input_context_sha256=m.core.digest(c))
        if self.fault=='after_local_AD':raise RuntimeError('local_AD_primary')
        assert actual==nominal
        std=np.tile(np.asarray(t['action_std'],np.float32),5).astype(float)
        j=(1+.05*c['latent_history'][-1][0])*self.b/std
        for _ in range(192):meter.call('vjp_directions',lambda:None)
        return dict(input_context_sha256='0'*64 if self.fault=='wrong_context' else m.core.digest(c),
            task_id=m.core.task_id(t),at_logits=[0.]*10,terminal=actual['latent_history'][-1],jacobian=j.tolist())


@pytest.fixture(scope='module')
def complete():
    events=[];backend=ArrayBackend();r=m.run_task(task(),backend,events.append)
    return r,events,backend


def test_genuine_all_slots_and_exact_all_stage_budget(complete):
    r,events,b=complete
    assert r['status']=='witness_completed_not_admitted' and r['error'] is None and b.closed
    assert len(r['candidates'])==64 and len(r['prefix_compilations'])==16
    for phase,caps in m.budgets().items():
        for resource,n in caps.items():assert r['resources'][phase][resource]==dict(attempted=n,completed=n)
    assert sum(r['resources'][p]['predict']['completed'] for p in m.budgets())==126
    assert len([e for e in events if e['kind']=='candidate_complete'])==64
    assert not r['P1_admission'] and not r['model_task_eligible']


def test_suffix_is_recompiled_in_every_actual_own_context_not_rootJ(complete):
    r,_,b=complete
    assert len(b.suffix_contexts)==16
    assert len({m.core.digest(c) for c in b.suffix_contexts})>1
    derivatives=r['prefix_compilations']
    for i,d in enumerate(derivatives):
        assert d['input_context_sha256']==m.core.digest(r['candidates'][4*i]['contexts'][0])
        assert d['derivative']['input_context_sha256']==m.core.digest(b.suffix_contexts[i])
    assert any(d['derivative']['jacobian']!=derivatives[0]['derivative']['jacobian'] for d in derivatives[1:])


def test_original_Q_goal_actions_and_independent_winner_replays(complete):
    r,_,_=complete;t=task();w=r['witness'];m.core.check_prediction(t,w)
    best=min(r['candidates'],key=lambda c:c['cost'])
    assert r['winner']==best and w['contexts']==best['contexts'] and w['cost']==best['cost']
    for c in r['candidates']:
        mid,end=c['contexts']
        assert c['commands']==m.edge(t,t['root_context'],c['logits'][0],mid)+m.edge(t,mid,c['logits'][1],end)
        assert c['cost']==m.core.model_cost(t,end['latent_history'][-1])


def test_shared_prefix_costs_and_no_zero_suffix_padding_or_cache(complete):
    _,_,b=complete
    assert sum(label=='actual_prefix' for _,_,label in b.calls)==16
    assert sum(label=='suffix_nominal' for _,_,label in b.calls)==16
    assert sum(label=='local_AD' for _,_,label in b.calls)==16
    assert sum(label.startswith('candidate_') for _,_,label in b.calls)==64
    assert sum(phase=='witness' for phase,_,_ in b.calls)==6


def test_core_compatible_producer_witness_journal(complete):
    r,_,_=complete
    p=dict(budgets=dict(direct={k:0 for k in m.core.RESOURCES},**{
        phase:{k:v[k] for k in m.core.RESOURCES} for phase,v in m.budgets().items()}))
    receipt=m.core.ledger(p,r['journal'],False)
    assert receipt['producer']['predict']['completed']==120
    assert receipt['witness']['predict']['completed']==6
    assert {e['evaluation_id'] for e in r['journal']}=={'producer','witness'}


def test_first_tie_stable_and_duplicate_slots_paid():
    t=task();t['goal']=[0.]*192;r=m.run_task(t,ArrayBackend(),lambda e:None)
    assert r['error'] is None and r['winner']['index']==0
    assert len(r['candidates'])==64 and r['resources']['producer']['selection']['completed']==64
    assert r['candidates'][0]['commands']==r['candidates'][1]['commands']


@pytest.mark.parametrize('fault',['prefix','after_local_AD','wrong_context'])
def test_faults_preserve_partial_nonrefundable_actual_work(fault):
    b=ArrayBackend(fault);events=[];r=m.run_task(task(),b,events.append)
    assert r['status']=='failed' and r['error'] and r['witness'] is None and b.closed
    if fault=='prefix':assert any(e['resource']=='predict' and not e['completed'] for e in r['journal'])
    else:
        assert any(e['kind']=='suffix_AD_forward' for e in events)
        if fault=='after_local_AD':assert any(e['resource']=='jacobian' and not e['completed'] for e in r['journal'])
    assert r['resources']['witness']['predict']['attempted']==0


def test_primary_not_overridden_by_finish_journal_error():
    def sink(e):
        if e['kind']=='producer_finished':raise OSError('secondary_sink')
    r=m.run_task(task(),ArrayBackend('prefix'),sink)
    assert r['error']['message']=='first_prefix_error' and r['secondary_errors'][0]['message']=='secondary_sink'


def test_attempt_sink_failure_stops_before_native_call():
    b=ArrayBackend()
    def sink(e):
        if e['kind']=='operation_attempted':raise OSError('attempt_disk_error')
    r=m.run_task(task(),b,sink)
    assert r['status']=='failed' and b.calls==[]
    assert r['resources']['producer']['action_encoder']==dict(attempted=1,completed=0)


def test_phase_specific_predispatch_caps_not_only_total():
    meter=m.Meter(lambda e:None);meter.phase='witness'
    for _ in range(6):meter.call('predict',lambda:None)
    called=[]
    with pytest.raises(ValueError):meter.call('predict',lambda:called.append(1))
    with pytest.raises(ValueError):meter.call('jacobian',lambda:called.append(2))
    assert called==[]


@pytest.mark.parametrize('cap',[1., float(np.sqrt(10)*.75)])
def test_wrong_packet_ball_rejected_before_backend(cap):
    t=task();t['packet_logit_cap']=cap;b=ArrayBackend()
    with pytest.raises(ValueError):m.run_task(t,b,lambda e:None)
    assert b.calls==[]


def test_bounded_ridge_is_one_svd_and_one_solve_and_matches_normal_equation():
    t=task();meter=m.Meter(lambda e:None);j=np.zeros((192,10));j[:10]=np.eye(10)
    result=m.compile_response(t,j,np.zeros(192),t['packet_logit_cap'],meter)
    a=j/np.sqrt(192);r=np.asarray(t['goal'])/np.sqrt(192)
    expected=np.linalg.solve(a.T@a+.002*np.eye(10),a.T@r)
    assert np.allclose(result['step'],expected,rtol=1e-6)
    assert sum(e['resource']=='svd' for e in meter.extended)==1
    assert sum(e['resource']=='linear_solve' for e in meter.extended)==1


def test_bounded_solver_records_finite_cpu_bisection_not_extra_svd():
    t=task();t['goal']=[100.]*192;meter=m.Meter(lambda e:None)
    j=np.zeros((192,10));j[:10]=np.eye(10)
    result=m.compile_response(t,j,np.zeros(192),t['packet_logit_cap'],meter)
    assert np.linalg.norm(result['step'])<=t['packet_logit_cap']
    assert result['arithmetic']['bisection_iterations']==70
    assert result['ball_multiplier']>0


def test_stage_local_J_is_analytic_full192x10_with_both_goal_and_context_pin(complete):
    r,_,b=complete;d=r['prefix_compilations'][5]['derivative'];c=b.suffix_contexts[5]
    expected=(1+.05*c['latent_history'][-1][0])*b.b/np.tile(np.asarray(task()['action_std'],np.float32),5)
    assert np.array_equal(d['jacobian'],expected)
    assert d['task_id']==m.core.task_id(task()) and np.asarray(d['jacobian']).shape==(192,10)


def test_store_exclusive_durable_and_global_reserve(tmp_path,monkeypatch):
    s=m.Store(tmp_path/'new');sink=s.sink('root.events.jsonl');sink(dict(kind='test',array=[1,2]))
    s.put('report.json',dict(status='mock_only'))
    assert [json.loads(x)['kind'] for x in (s.output/'root.events.jsonl').read_text().splitlines()]==['journal_created','test']
    with pytest.raises(FileExistsError):s.put('report.json',{})
    with pytest.raises(ValueError):m.Store(s.output)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=m.RESERVE))
    with pytest.raises(ValueError):s.put('later.json',{})
    assert not (s.output/'later.json').exists()


def test_global_output_cap_includes_final_reports(tmp_path):
    s=m.Store(tmp_path/'new');s.used=m.OUTPUT_LIMIT
    with pytest.raises(ValueError):s.put('report.json',{})
    assert not (s.output/'report.json').exists()


def test_run_requires_explicit_host_authority_before_verify_or_import(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('verify must not start'))
    with pytest.raises(ValueError):m.run(tmp_path/'notthere','0'*64,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_native_rejects_cpu_before_model_import_and_collect_does_not_open_gains():
    with pytest.raises(ValueError):m.NativeCuda(None,SimpleNamespace(type='cpu'),task())
    text=inspect.getsource(m.collect)
    assert "r['actual_direct_resources']" in text
    assert 'direct_runs' not in text and 'root_0.json' not in text and 'io.collect(' not in text
    assert 'suffix_jacobian' in inspect.getsource(m.run_task)
    assert 'requires_grad_(True)' in inspect.getsource(m.NativeCuda.suffix_jacobian)


def test_protocol_has_no_tau_or_posthoc_selection():
    assert m.PROTOCOL['prefix_slots']==16 and m.PROTOCOL['candidates']==64
    assert not any('tau' in key or 'epsilon' in key for key in m.PROTOCOL)
    assert m.PROTOCOL['P1_admission'] is False


def mocked_inputs():
    return dict(schema=m.NAME+'.inputs',direct_prepared='/mock/direct',direct_prepare_sha256='a'*64,
        task_directory='/mock/tasks',bindings={},tasks=[dict(root_sha256='1'*64,task_id=m.core.task_id(task()),task=task()),
        dict(root_sha256='2'*64,task_id=m.core.task_id(task()),task=task())],policy='/mock/local_only',
        model_bindings={},inherited=dict(mock=True),source_purpose='calibration_exploration',
        history_semantics='causal_padding_not_observed_transitions')


def test_prepare_verify_seal_roundtrip_and_external_sha(tmp_path,monkeypatch):
    inputs=mocked_inputs()
    monkeypatch.setattr(m,'collect',lambda *a:deepcopy(inputs))
    monkeypatch.setattr(m.io,'source_files',lambda:[])
    out=tmp_path/'prepared'
    receipt=m.prepare('/mock/direct','a'*64,'/mock/tasks',out)
    assert receipt['status']=='prepared_not_executed'
    sha=m.io.sha(out/'prepare.json');p=m.verify(out,sha)
    assert p['algorithm']==m.PROTOCOL and p['budget_per_root']==m.budgets()
    with pytest.raises(ValueError):m.verify(out,'0'*64)
    with pytest.raises(ValueError):m.prepare('/mock/direct','a'*64,'/mock/tasks',out)


def test_verify_reconstructs_inputs_not_only_file_hashes(tmp_path,monkeypatch):
    inputs=mocked_inputs();monkeypatch.setattr(m,'collect',lambda *a:deepcopy(inputs))
    monkeypatch.setattr(m.io,'source_files',lambda:[])
    out=tmp_path/'prepared';m.prepare('/mock/direct','a'*64,'/mock/tasks',out)
    sha=m.io.sha(out/'prepare.json');inputs['history_semantics']='invented_history'
    with pytest.raises(ValueError,match='input reconstruction'):m.verify(out,sha)


def test_run_loader_failure_keeps_primary_through_sync_and_final_verify(tmp_path,monkeypatch):
    # Stub modules ONLY. There is no real torch import, CUDA allocation, model,
    # predictor or encoder in this fault test.
    import sys
    import types
    torch=types.ModuleType('torch')
    torch.backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)))
    torch.device=lambda value:SimpleNamespace(type='cuda')
    def sync():raise OSError('sync_secondary')
    torch.cuda=SimpleNamespace(synchronize=sync)
    attention=types.ModuleType('torch.nn.attention')
    attention.SDPBackend=SimpleNamespace(MATH='mock')
    @contextmanager
    def kernel(value):yield
    attention.sdpa_kernel=kernel
    loader=types.ModuleType('experiments.checkpoint_compat')
    def fail_load(*a,**k):raise RuntimeError('loader_primary')
    loader.load_compatible_pretrained=fail_load
    gpu=types.ModuleType('experiments.gpu_preflight');gpu.require_cuda_allocation=lambda device:None
    for name,value in [('torch',torch),('torch.nn.attention',attention),
        ('experiments.checkpoint_compat',loader),('experiments.gpu_preflight',gpu)]:
        monkeypatch.setitem(sys.modules,name,value)
    calls=[]
    def verify(*a):
        calls.append(1)
        if len(calls)>1:raise ValueError('source_secondary')
        return dict(inputs=mocked_inputs())
    monkeypatch.setattr(m,'verify',verify)
    result=m.run(tmp_path/'prepared','a'*64,tmp_path/'out',authorize_host_cuda=True)
    assert result['status']=='partial_or_failed' and result['roots']==[]
    assert [e['message'] for e in result['errors']]==['loader_primary','sync_secondary','source_secondary']
    assert (tmp_path/'out'/'report.json').is_file()


def test_run_never_establishes_direct_stall(complete):
    assert complete[0]['direct_plateau_established_by_this_producer'] is False
    assert 'direct finite-budget stall evidence' in complete[0]['missing']
