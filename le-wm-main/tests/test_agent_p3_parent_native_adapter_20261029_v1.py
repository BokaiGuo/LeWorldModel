"""Explicit numpy dynamical callbacks, not a model or physical experiment."""
from copy import deepcopy
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_parent_native_adapter_20261029_v1 as m


class Sink:
    def __init__(self):self.events=[]
    def event(self,v):self.events.append(deepcopy(v))


def task():
    context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,
        action_dim=10,latent_history=np.zeros((3,192),np.float32).tolist(),action_history=np.zeros((2,10),np.float32).tolist())
    t=dict(root_context=context,goal=[1.,.5]+[0.]*190,projection=np.eye(192).tolist(),
        action_mean=[.125,-.25],action_std=[2.,4.])
    t['binding']={k:m.digest(k) for k in m.core.BINDINGS}
    t['binding'].update(goal_sha256=m.digest(t['goal']),projection_sha256=m.digest(t['projection']),
        context_sha256=m.digest(context),normalizer_sha256=m.digest([t['action_mean'],t['action_std']]),
        action_domain_sha256=m.digest(m.core.action_domain()))
    return m.core.validate_task(t)


class RAM:
    """Non-neural linear action transition with an autonomous clock coordinate.

    Clock0: short cannot affect y, later tokens can. Real predicted child has
    nonzero clock and its short/long spans coincide -> original child stop.
    J is computed analytically for this callback, never reused from the root.
    """
    kind='RAM_FIXTURE'
    def __init__(self,t,s,fail=False):self.task=deepcopy(t);self.sink=s;self.signals=[];self.replays=[];self.fail=fail
    def replay(self,parent,actions,segmented,ledger):
        self.replays.append((deepcopy(parent),deepcopy(actions),segmented));c=deepcopy(parent);path=[]
        for raw in np.asarray(actions,np.float32).reshape(-1,5,2):
            token=((raw-np.asarray(self.task['action_mean'],np.float32))/np.asarray(self.task['action_std'],np.float32)).reshape(10)
            outgoing=np.asarray(c['action_history']+[token.tolist()],np.float32)[None]
            encoded=ledger.call('action_encoder',lambda:np.repeat(outgoing.mean(axis=2,keepdims=True),192,axis=2).astype(np.float32),shape=[1,3,10])
            state=np.asarray(c['latent_history'],np.float32)[None]
            def predict():
                def transition():
                    if self.fail:raise RuntimeError('RAM nested predictor failure')
                    out=state.copy();v=out[0,-1];clock=float(v[2]);a=np.float32(1+clock*1e8);b=np.float32(clock>=5e-9)
                    v[0]+=a*raw[:,0].sum(dtype=np.float32);v[1]+=b*raw[:,1].sum(dtype=np.float32);v[2]+=np.float32(1e-8)
                    return out
                z=ledger.call('predictor',transition,shape=[1,3,192])
                return ledger.call('pred_proj',lambda:z.reshape(3,192).copy(),shape=[3,192]).reshape(1,3,192)
            z=ledger.call('predict',predict,shape=[1,3,192])
            c=dict(c,latent_history=c['latent_history'][1:]+[z[0,-1].tolist()],action_history=[c['action_history'][-1],token.tolist()])
            path.append(deepcopy(c))
        return path
    def signal(self,req,ledger):
        assert req['binding']==self.task['binding'] and req['parent_context']==self.task['root_context']
        self.signals.append(deepcopy(req));h=req['horizon'];zero=np.zeros((h*5,2),np.float32)
        nominal=self.replay(req['parent_context'],zero,False,ledger)[-1]['latent_history'][-1]
        def jacobian():
            ad=self.replay(req['parent_context'],zero,False,ledger)[-1]['latent_history'][-1]
            assert ad==nominal
            j=np.zeros((192,10*h),np.float32);clock=float(req['parent_context']['latent_history'][-1][2])
            for k in range(h):
                j[0,10*k:10*k+10:2]=np.float32(1+clock*1e8)
                j[1,10*k+1:10*k+10:2]=np.float32(clock>=5e-9);clock=float(np.float32(clock)+np.float32(1e-8))
            for row in j:
                ledger.call('vjp',lambda row=row:ledger.call('grad',lambda:ledger.call('raw_backward',lambda:row.copy())),directions=1)
            return j
        j=ledger.call('jacobian',jacobian)
        return dict(binding_sha256=req['binding_sha256'],parent_context_sha256=req['parent_context_sha256'],
            horizon=h,nominal=nominal,jacobian=j.tolist(),producer_sha256=m.digest('RAM exact transition/J callback'))


def run(mode='adaptive',**kwargs):
    sink=Sink();backends=[]
    def factory(t,s):
        b=RAM(t,s);backends.append(b);return b
    result=m.run_core(task(),factory,sink,mode=mode,permit_ram=True,**kwargs)
    return result,sink,backends


@pytest.fixture(scope='module')
def adaptive():return run()


@pytest.fixture(scope='module')
def fixed():return run('fixed')


def test_actual_child_stop_from_new_full_parent_signals(adaptive):
    r,sink,backends=adaptive
    assert not r['errors'],r['errors']
    assert r['model_plan'] and r['realized_progress_depth']==1 and r['realized_widths']==[4]
    assert r['queues'][0]['decision']['horizon']==3 and r['queues'][0]['decision']['program_depth_limit']==3
    children=r['queues'][1:];assert len(children)==4
    assert all(q['status']=='stopped_by_original_child_policy' and q['decision']['program_type']=='composition'
        and q['decision']['extension_type']=='progress' and q['decision']['program_depth_limit']==1 for q in children)
    assert len(r['signal_mappings'])==10 and len(backends)==5
    rootj=r['signal_mappings'][0]['native_response']['jacobian']
    assert any(x['native_response']['jacobian']!=rootj for x in r['signal_mappings'][2:] if x['native_response']['horizon']==1)
    for x in r['signal_mappings']:
        assert x['root_binding']==task()['binding'] and x['root_task_sha256']==m.digest(task())
        assert x['native_request']['binding']==x['parent_task']['binding']
        assert x['parent_task']['root_context']==x['outer_request']['parent_context']
        assert x['native_response']['binding_sha256']==m.digest(x['parent_task']['binding'])
        assert x['outward_response']['binding_sha256']==m.digest(task()['binding'])
    assert r['scalar_backward_calls'] is r['AD_primal_predict_API'] is r['Jacobian_elements'] is None


def test_fixed_real_depth3_and_exact_composition_chain(fixed):
    r,sink,backends=fixed;assert not r['errors'],r['errors']
    assert r['realized_progress_depth']==3 and len(r['jobs'])==144
    assert len([x for x in r['jobs'] if x['status']=='returned'])==36
    assert r['resource_receipt']['completed']['predict_API']<=114
    assert all(q['decision']['candidates']==4 and q['decision']['program_depth_limit']==3 for q in r['queues'])
    for row in r['jobs']:
        if row['node'] is None:continue
        node=deepcopy(row['node']);node.pop('cost');m.validate_parent(task(),node)
        assert len(node['context']['latent_history'])==3 and len(node['context']['action_history'])==2
        assert node['actions']==[a for e in node['edges'] for a in e['actions']]
    assert r['replays'][0]['path']==r['replays'][1]['path']==r['model_plan']['path']
    assert all(r[k] is False for k in m.FLAGS)


def test_root_unchanged_and_dual_identity_not_fake_native(adaptive):
    r,_,_=adaptive
    child=next(x for x in r['signal_mappings'] if x['parent_task']['root_context']!=task()['root_context'])
    assert child['native_request']['binding_sha256']!=child['outer_request']['binding_sha256']
    assert child['native_response']['producer_sha256']!=child['outward_response']['producer_sha256']
    assert child['native_response']['jacobian']==child['outward_response']['jacobian']
    assert child['mapping_sha256']==m.digest({k:v for k,v in child.items() if k!='mapping_sha256'})


def test_atomic_expansion_cap_before_any_program():
    r,_,backends=run(max_requests=3)
    assert r['status']=='expansion_cap_before_node_queue' and not r['errors']
    assert len(r['queues'][0]['requests'])==4 and all(x['status']=='not_requested' for x in r['jobs'])
    assert r['resource_receipt']['attempted']['program_compile_calls']==0 and r['model_plan'] is None


def test_atomic_resource_cap_reserves_complete_queue_and_verify():
    c=m.default_caps('adaptive');c['predict_API']=19
    r,_,_=run(caps=c)
    assert r['status']=='resource_cap_before_node_queue' and not r['errors']
    assert r['resource_receipt']['attempted']['predict_API']==8
    assert r['resource_receipt']['attempted']['program_compile_calls']==0 and r['model_plan'] is None


def test_R_original8_requests_entirely_refused_no_fallback():
    t=task();t['goal']=[1.]*192;t['binding']['goal_sha256']=m.digest(t['goal']);sink=Sink();backends=[]
    def factory(task,sink):
        b=RAM(task,sink);backends.append(b);return b
    r=m.run_core(t,factory,sink,mode='adaptive',permit_ram=True)
    assert r['status']=='refused_unsupported_original_type' and r['model_plan'] is None
    assert r['queues'][0]['decision']['program_type']=='reconfiguration'
    assert len(r['queues'][0]['requests'])==8 and sum(x['status']=='refused_unsupported_original_type' for x in r['jobs'])==8
    assert r['resource_receipt']['completed']['program_compile_calls']==0
    assert r['resource_receipt']['completed']['predict_API']==8


def test_queue_preserves_all8_without_claiming_policy_progress8(adaptive):
    r,_,_=adaptive;t=task();parent=m.root_node(t);q=r['queues'][0]
    d=dict(q['decision'],candidates=8);ledger=m.core.Ledger(m.default_caps('adaptive'))
    queue=m.atomic_queue(t,parent,d,r['signals'][0],0,144,ledger)
    assert queue['status']=='queued' and [x['candidate_index'] for x in queue['requests']]==list(range(8))
    # This tests the generic queue only: original adaptive progress selects4, R8 refuses.
    assert r['adaptive_width8_executed'] is False


def test_generic8_bank_all_callbacks_and_no_partial_assignment():
    t=task();sink=Sink();adapter=m.ParentNativeAdapter(t,lambda t,s:RAM(t,s),sink,permit_ram=True)
    ledger=m.core.Ledger(m.default_caps('adaptive'));ledger.scope=dict(test='generic8')
    node=m.root_node(t)
    signals,receipt=m.core.local_signal(t,node,adapter,ledger,m.core.policy(t))
    original=m.core.choose(t,node,signals,'adaptive')
    assert original['candidates']==4  # generic bank support is NOT original progress8
    decision=dict(original,candidates=8)
    queue=m.atomic_queue(t,node,decision,receipt,0,144,ledger)
    assert queue['status']=='queued' and len(queue['requests'])==8
    results=[ledger.call('program_compile',lambda req=req:adapter.generate(req,ledger)) for req in queue['requests']]
    assert len(results)==8 and ledger.completed['program_compile_calls']==8
    assert ledger.completed['solve_calls']==1 and ledger.completed['predict_API']==8+8*3
    for result in results:m.core.checked_path(t,t['root_context'],result['actions'],result['path'])
    assert results[0]['actions']==np.zeros((15,2),np.float32).tolist()
    assert results[-1]['actions']!=results[0]['actions']
    # One fewer free slot must reject the ENTIRE same8 queue.
    refused=m.atomic_queue(t,node,decision,receipt,137,144,ledger)
    assert refused['status']=='expansion_cap_before_node_queue' and len(refused['requests'])==8


@pytest.mark.parametrize('field',('context','actions','path','edges','indices'))
def test_complete_parent_chain_mutations_rejected(fixed,field):
    r,_,_=fixed;node=deepcopy(next(x['node'] for x in r['jobs'] if x['node'] and x['node']['depth']==2))
    node.pop('cost')
    if field=='context':node[field]['latent_history'][0][0]+=.1
    elif field=='actions':node[field][0][0]+=.1
    elif field=='path':node[field][0]['action_history'][0][0]+=.1
    elif field=='edges':node[field][0]['output_context_sha256']='f'*64
    else:node[field].pop()
    with pytest.raises(ValueError):m.validate_parent(task(),node)


@pytest.mark.parametrize('damage',('root','parent','raw','older','edge','future','state'))
def test_bad_bindings_context_or_leak_reject(adaptive,damage):
    r,_,_=adaptive;mapping=deepcopy(r['signal_mappings'][0]);req=mapping['outer_request'];sink=Sink()
    adapter=m.ParentNativeAdapter(task(),lambda t,s:RAM(t,s),sink,permit_ram=True)
    if damage=='root':req['binding']['model_sha256']='f'*64
    elif damage=='parent':req['parent_context_sha256']='f'*64
    elif damage in ('future','state'):req[damage]=[0]
    elif damage=='older':req['parent_context']['latent_history'][0][0]=.5
    elif damage=='raw':req['parent_context']['action_history'][0][0]=.5
    else:req['operation']='verify_program'
    ledger=m.core.Ledger(m.default_caps('adaptive'));ledger.scope=dict(test=True)
    with pytest.raises(ValueError):adapter.signal(req,ledger)


def test_stale_root_J_cannot_generate_child(adaptive):
    r,_,_=adaptive;t=task();sink=Sink();adapter=m.ParentNativeAdapter(t,lambda t,s:RAM(t,s),sink,permit_ram=True)
    ledger=m.core.Ledger(m.default_caps('adaptive'));ledger.scope=dict(test=True)
    req=deepcopy(r['jobs'][0]['request'])
    # No corresponding paid parent responses exist in this fresh adapter.
    with pytest.raises(ValueError):adapter.generate(req,ledger)


def test_nested_failure_no_refunds_and_no_partial_plan():
    sink=Sink()
    r=m.run_core(task(),lambda t,s:RAM(t,s,fail=True),sink,mode='adaptive',permit_ram=True)
    assert r['status']=='failed_or_budget_exhausted' and r['model_plan'] is None and r['errors']
    c=r['resource_receipt'];assert c['attempted']['predict_API']==1 and c['completed']['predict_API']==0
    assert c['attempted']['predictor_calls']==1 and c['completed']['predictor_calls']==0
    assert any(e['kind']=='resource_failure' for e in sink.events)


def test_no_cpu_fallback_or_cold_foreign_import():
    code=f'import sys; import experiments.spectral_reachability.{m.NAME}; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules'
    subprocess.run([sys.executable,'-c',code],cwd=m.progress.ROOT,check=True)
    r=m.run_core(task(),lambda t,s:RAM(t,s),Sink(),mode='adaptive')
    assert r['status']=='failed_or_budget_exhausted' and r['resource_receipt']['attempted']['predict_API']==0
