"""RAM-only C3 contracts: no model, GPU, physics, actual prepare or outcomes."""
from copy import deepcopy
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as m


def h(v): return m.digest(str(v))


def task(slot=0):
    c=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=4,action_dim=10,
        latent_history=np.zeros((3,4)).tolist(),action_history=np.zeros((2,10)).tolist())
    t=dict(root_context=c,goal=[1.,1.,0.,0.],projection=np.eye(4).tolist(),action_mean=[0.,0.],action_std=[1.,1.])
    t['binding']={k:h((slot,k)) for k in m.BINDINGS}
    t['binding'].update(context_sha256=m.digest(c),goal_sha256=m.digest(t['goal']),projection_sha256=m.digest(t['projection']),
        normalizer_sha256=m.digest([t['action_mean'],t['action_std']]),action_domain_sha256=m.digest(m.action_domain()))
    return m.validate_task(t)


def caps(): return dict.fromkeys(m.AXES,500000)  # RAM counter limit, NOT a proposed model budget.


class Backend:
    kind='RAM_FIXTURE'
    def __init__(self,geometry='progress',fault=None,batch=1):
        self.geometry=geometry; self.fault=fault; self.batch=batch; self.requests=[]; self.forward_calls=0

    def forward(self,ledger):
        self.forward_calls+=1; b=self.batch
        ledger.call('action_encoder',lambda:None,shape=[b,3,10])
        def predict():
            ledger.call('predictor',lambda:None,shape=[b,3,4])
            ledger.call('pred_proj',lambda:None,shape=[3*b,4])
        ledger.call('predict',predict,shape=[b,3,4])

    def signal(self,req,ledger):
        self.requests.append(deepcopy(req)); horizon=req['horizon']
        for _ in range(horizon): self.forward(ledger)
        def jacobian():
            for _ in range(horizon): self.forward(ledger)
            ledger.call('vjp',lambda:ledger.call('grad',lambda:ledger.call('raw_backward',lambda:None)),directions=4)
        ledger.call('jacobian',jacobian)
        j=np.zeros((4,10*horizon))
        if self.geometry=='progress' or (self.geometry=='long' and horizon>1): j[:4,:4]=np.eye(4)
        if self.geometry=='signed_loss' and horizon==1: j[:4,:4]=np.eye(4)
        result=dict(binding_sha256=req['binding_sha256'],parent_context_sha256=req['parent_context_sha256'],
            horizon=horizon,nominal=deepcopy(req['parent_context']['latent_history'][-1]),jacobian=j.tolist(),producer_sha256=h('RAM_SIGNAL_ONLY'))
        if self.fault=='signal_outcome': result['native_success']=True
        if self.fault=='signal_parent': result['parent_context_sha256']=h('nearby not identical')
        return result

    def path(self,context,actions,ledger):
        out=[]; before=deepcopy(context)
        for k in range(len(actions)//5):
            self.forward(ledger)
            token=np.asarray(actions[5*k:5*k+5],np.float32).reshape(10).astype(float).tolist()
            last=np.asarray(before['latent_history'][-1],np.float32).copy()
            last[:2]+=np.asarray(actions[5*k],np.float32)
            after=dict(before,latent_history=before['latent_history'][1:]+[last.astype(float).tolist()],
                action_history=[before['action_history'][-1],token])
            out.append(after); before=after
        return out

    def generate(self,req,ledger):
        self.requests.append(deepcopy(req))
        require_no_oracle(req)
        raw=np.full((req['horizon']*5,2),np.float32((req['candidate_index']+1)/128)).astype(float).tolist()
        path=self.path(req['parent_context'],raw,ledger)
        result=dict(binding_sha256=req['binding_sha256'],parent_context_sha256=req['parent_context_sha256'],
            extension_type=req['extension_type'],actions=raw,path=path,producer_sha256=h('RAM_GENERATION_ONLY'))
        if self.fault=='program_parent': result['parent_context_sha256']=h('other node')
        if self.fault=='program_outcome': result['task_success']=False
        if self.fault=='program_kind': result['extension_type']='progress' if req['extension_type']=='reconfiguration' else 'reconfiguration'
        if self.fault=='program_history': result['path'][0]['action_history'][0][0]=1.
        if self.fault=='program_float': result['actions'][0][0]=.1
        if self.fault=='program_horizon': result['actions'].pop()
        if self.fault=='program_error': raise RuntimeError('RAM compiler failed after paid work')
        return result

    def replay(self,req,ledger):
        self.requests.append(deepcopy(req)); result=self.path(req['parent_context'],req['actions'],ledger)
        if self.fault=='replay': result[-1]['latent_history'][-1][0]+=1
        return result


def require_no_oracle(value):
    if isinstance(value,dict):
        assert not (set(value)&{'outcome','reward','success','native_success','physical_state','teacher_rank','cost','winner_index'})
        for v in value.values(): require_no_oracle(v)
    elif isinstance(value,list):
        for v in value: require_no_oracle(v)


@pytest.mark.parametrize('geometry,kind,horizon,depth',[('progress','progress',1,1),('reconfiguration','reconfiguration',1,3),('long','reconfiguration',3,3)])
def test_actual_same_task_signal_to_type_depth_horizon_and_witness_queue(geometry,kind,horizon,depth):
    t=task(); backend=Backend(geometry); result=m.run_task(t,backend,caps(),mode='adaptive',permit_ram=True)
    assert result['status']=='ram_complete_not_evidence' and result['model_plan'] is not None
    q=result['queues'][0]; assert q['decision']['extension_type']==kind
    assert q['decision']['horizon']==horizon and q['decision']['program_depth_limit']==depth
    assert len(q['requests'])==(4 if kind=='progress' else 8)
    assert all(result[k] is False for k in m.FLAGS)
    assert len(result['jobs'])==144 and len(result['replays'])==2
    assert backend.forward_calls==result['resource_receipt']['attempted']['predict_API']
    assert result['resource_receipt']['attempted']==result['resource_receipt']['completed']
    assert result['resource_receipt']['attempted']['array_SVD_calls']==2*len(result['signals'])
    if depth==3:
        composed=[q for q in result['queues'] if q['depth']==1]
        assert composed and all(q['decision']['program_type']=='composition' for q in composed)
        for q in composed:
            assert all(r['prefix_edges'] and r['prefix_actions'] for r in q['requests'])
        assert any(r['node']['depth']==3 for r in result['jobs'] if r['status']=='returned')
    assert t==task()  # Compiler received snapshots, not mutable task ownership.


def test_fixed_control_pays_signal_but_does_not_use_it_and_keeps_counter_axes():
    result=m.run_task(task(),Backend('long',batch=2),caps(),mode='fixed',permit_ram=True)
    assert result['model_plan'] is not None
    assert all(q['decision']['extension_type']=='progress' and q['decision']['horizon']==1 and q['decision']['candidates']==4 for q in result['queues'])
    r=result['resource_receipt']; n=r['attempted']['predict_API']
    assert r['attempted']['predict_samples']==2*n and r['attempted']['predict_context_tokens']==6*n
    assert r['attempted']['pred_proj_rows']==6*n and r['attempted']['array_SVD_calls']>0
    assert r['backward_engine_calls'] is r['FLOPs'] is r['cuda_ns'] is None


def test_signed_horizon_subspace_loss_kept_and_not_cost_stagnation():
    t=task(); ledger=m.Ledger(caps()); ledger.scope={'phase':'RAM'}
    node=dict(context=t['root_context'],depth=0)
    s,r=m.local_signal(t,node,Backend('signed_loss'),ledger,m.policy(t))
    assert r['signed_horizon_coverage_gain']==-1. and s.horizon_gain==0.
    assert s.unreachable_fraction==0. and m.choose(t,node,s,'adaptive')['horizon']==1
    r['spectra'][0]['unreachable_fraction_same_reference_residual']=.9
    # A changed diagnostic object cannot mutate immutable scheduler scalars.
    assert s.unreachable_fraction==0.


@pytest.mark.parametrize('fault',['signal_outcome','signal_parent','program_parent','program_outcome','program_kind',
    'program_history','program_float','program_horizon','program_error','replay'])
def test_failclosed_leak_identity_horizon_and_replay_mutations(fault):
    backend=Backend('progress',fault=fault); r=m.run_task(task(),backend,caps(),mode='adaptive',permit_ram=True)
    assert r['model_plan'] is None and r['errors'] and r['status']=='failed_or_budget_exhausted'
    assert len(r['jobs'])==144 and len(r['replays'])==2
    assert r['resource_receipt']['attempted']['predict_API']>0
    m.audit_ledger(r['resource_receipt'])
    if fault.startswith('program'):
        assert sum(j['status']=='failed' for j in r['jobs'])==1
        assert sum(j['status']=='blocked_after_failure' for j in r['jobs'])==3
    if fault=='replay': assert r['diagnostic_best'] is not None  # Preserve best but never release.


@pytest.mark.parametrize('field',['reward','physical_state','teacher_rank'])
def test_task_oracle_fields_rejected_before_backend(field):
    t=task(); t[field]=0; backend=Backend()
    with pytest.raises(ValueError): m.run_task(t,backend,caps(),mode='fixed',permit_ram=True)
    assert backend.requests==[]


def test_other_root_signal_cannot_drive_this_task():
    t=task(); other=task(1); p=m.policy(t)
    with pytest.raises(ValueError): m.choose(t,dict(depth=0,context=t['root_context']),m.FrozenSignals(m.policy(other).source_seal,.9,.9),'adaptive')
    assert p.source_seal!=m.policy(other).source_seal


def test_same_task_stale_parent_signal_cannot_drive_child_queue():
    t=task(); child=dict(depth=1,context=deepcopy(t['root_context']))
    child['context']['latent_history'][-1][0]=.5
    stale=m.FrozenSignals(m.policy(t).source_seal,.9,.9)
    with pytest.raises(ValueError): m.choose(t,child,stale,'adaptive')


def test_nonrefundable_budget_failure_and_complete_paired_denominator():
    cap=caps(); cap['predict_API']=1
    result=m.run_panel([task(),task(1)],lambda t,mode:Backend(),cap,permit_ram=True)
    assert result['fixed_denominator']==4 and len(result['rows'])==4 and result['failed_or_incomplete']==4
    for row in result['rows']:
        r=row['runtime']; assert r['model_plan'] is None
        assert r['resource_receipt']['attempted']['predict_API']==1
        assert any(e['status']=='blocked_before_dispatch' for e in r['resource_receipt']['events'])
        m.audit_ledger(r['resource_receipt'])


def test_setup_failure_still_retains_every_task_method():
    def broken(t,mode): raise RuntimeError('RAM setup failure')
    r=m.run_panel([task(),task(1)],broken,caps(),permit_ram=True)
    assert r['fixed_denominator']==r['failed_or_incomplete']==4 and all(x['runtime'] is None for x in r['rows'])


def test_counters_missing_null_bool_and_forged_delta_rejected():
    for v in (None,True,-1):
        cap=caps(); cap['grad_API']=v
        with pytest.raises(ValueError): m.Ledger(cap)
    cap=caps(); cap.pop('solve_calls')
    with pytest.raises(ValueError): m.Ledger(cap)
    ledger=m.Ledger(caps()); ledger.scope={'phase':'RAM'}; ledger.call('predict',lambda:None,shape=[2,3,4])
    r=ledger.receipt(); m.audit_ledger(r)
    r['events'][0]['delta']['predict_samples']=1
    with pytest.raises(ValueError): m.audit_ledger(r)


def test_failed_svd_attempt_is_paid_and_never_turned_into_signal(monkeypatch):
    ledger=m.Ledger(caps()); ledger.scope={'phase':'RAM'}; t=task()
    def fail(*a,**kw): raise RuntimeError('RAM SVD failure')
    monkeypatch.setattr(m.np.linalg,'svd',fail)
    with pytest.raises(RuntimeError): m.local_signal(t,dict(context=t['root_context']),Backend(),ledger,m.policy(t))
    r=ledger.receipt(); assert r['attempted']['array_SVD_calls']==1 and r['completed']['array_SVD_calls']==0
    assert r['attempted']['jacobian_API']==2; m.audit_ledger(r)


def test_shapes_and_physics_not_accepted_as_hidden_zero_work():
    for resource,shape in [('predict',[1,2,4]),('action_encoder',[True,3,10]),('physics',[1,3,4])]:
        with pytest.raises(ValueError): m.delta(resource,shape)
    ledger=m.Ledger(caps()); ledger.scope={'phase':'RAM'}
    ledger.call('encoder',lambda:None,shape=[2,3,224,224]); ledger.call('projector',lambda:None,shape=[2,192])
    ledger.call('solve',lambda:None); ledger.call('model_svd',lambda:None)
    r=ledger.receipt(); assert r['attempted']['encoder_samples']==2 and r['attempted']['projector_calls']==1
    assert r['attempted']['solve_calls']==r['attempted']['model_SVD_calls']==1; m.audit_ledger(r)


def test_ram_requires_explicit_optin_no_runtime_imports():
    with pytest.raises(ValueError): m.run_task(task(),Backend(),caps(),mode='adaptive')
    assert not any(x in sys.modules for x in ('torch','gymnasium','pymunk'))


def test_external_prepare_SHA_and_full_rebuild(monkeypatch):
    value=dict(schema='RAM prepare',tasks=[task()],resource_caps=None)
    monkeypatch.setattr(m,'prepare_material',lambda:deepcopy(value))
    monkeypatch.setattr(m,'sha',lambda _:h('RAM source'))
    monkeypatch.setattr(m,'read',lambda _:deepcopy(value))
    assert m.verify('/RAM/no file',h('RAM source'))==value
    with pytest.raises(ValueError): m.verify('/RAM/no file',h('other'))
    monkeypatch.setattr(m,'read',lambda _:{**value,'resource_caps':{}})
    with pytest.raises(ValueError): m.verify('/RAM/no file',h('RAM source'))


def test_native_adapter_requires_durable_sink_before_dispatch():
    backend=Backend(); backend.kind='HOST_CUDA'
    with pytest.raises(ValueError,match='durable event sink'):
        m.run_task(task(),backend,caps(),mode='adaptive')
    assert backend.requests==[]


def test_predispatch_journal_failure_paid_no_delegate_and_first_error_retained():
    calls=[]
    def sink(e): raise OSError('RAM journal failure')
    ledger=m.Ledger(caps(),sink=sink); ledger.scope={'phase':'RAM'}
    with pytest.raises(OSError,match='RAM journal failure'):
        ledger.call('predict',lambda:calls.append(1),shape=[1,3,4])
    r=ledger.receipt(); assert calls==[] and r['attempted']['predict_API']==1 and r['completed']['predict_API']==0
    assert 'failure journal' in r['events'][0]['error']; m.audit_ledger(r)


def test_sink_order_and_nested_inclusive_time_not_additive_global():
    events=[]; ticks=iter(range(100)); ledger=m.Ledger(caps(),clock=lambda:next(ticks),sink=events.append)
    ledger.scope={'phase':'RAM'}
    ledger.call('predict',lambda:ledger.call('predictor',lambda:None,shape=[2,3,4]),shape=[2,3,4])
    assert [e['kind'] for e in events]==['resource_attempt','resource_attempt','resource_return','resource_return']
    r=ledger.receipt(); assert r['inclusive_operation_wall_ns']['predict']>r['inclusive_operation_wall_ns']['predictor']
    m.audit_ledger(r)
