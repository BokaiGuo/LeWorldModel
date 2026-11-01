"""Explicit NumPy/RAM native-interface doubles. No torch model or environment."""
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_feedback_gradient_mpc20_20261028_v1 as m
import agent_p3_same_feedback_physical20_20261028_v1_tests as p3test
import test_p1_constructive_gradient_mpc_cuda_20261027_v1 as gradtest


class Events:
    def __init__(self):self.events=[]
    def event(self,e):self.events.append(deepcopy(e))


def request():
    base,old=p3test.request();identity={k:old[k] for k in ('episode_id','source_slot','mode','boundary','raw10','observation')}
    r=dict(schema=m.NAME+'.node',base_task_sha256=m.digest(base),task=m.local_task(base,old['task']['root_context'],identity),**identity)
    template=gradtest.task(2026)
    for k in ('root_context','goal','projection','action_mean','action_std'):template[k]=deepcopy(base[k])
    template['root_context']=deepcopy(r['task']['root_context'])
    template['root_raw_action_history']=deepcopy(r['raw10'])
    m.gradient.core.validate_task(template);return base,r,template


class RAMGradient(gradtest.Backend):
    """Synthetic recurrence with complete native ABI transcripts, NOT a model."""
    def __init__(self,ledger,journal):
        super().__init__();self.ledger,self.journal=ledger,journal;self.serial=0
    def arrays(self,kind,operation_index=None,**values):
        self.journal.event(dict(kind=kind,scope=deepcopy(self.ledger.scope),phase=self.phase,step=self.step,
            operation_index=operation_index,arrays={k:m.signal.blob(v) for k,v in values.items()}))
    def call(self,resource,fn,inputs,shape=None):
        index=len(self.ledger.events)
        def actual():
            self.arrays('io_input',index,**{resource+':'+k:v for k,v in inputs.items()})
            out=fn();self.arrays('io_return',index,**{resource+':output':out});return out
        return self.ledger.call(resource,actual,shape=shape)
    def forwards(self,t,raw,meter,kind,mode=None):
        path=gradtest.path(t,raw);self.phase=f'forward_{self.serial:03}';self.serial+=1
        mode='gradient' if kind=='gradient_forward_packet' else mode
        self.ledger.scope=dict(forward=self.phase,group=meter.group,mode=mode)
        self.journal.event(dict(kind='forward_started',phase=self.phase,group=meter.group,mode=mode))
        latent=np.asarray(t['root_context']['latent_history'],np.float32)[None]
        history=np.asarray(t['root_context']['action_history'],np.float32)[None]
        mean,std=(np.asarray(t[k],np.float32) for k in ('action_mean','action_std'))
        for k,c in enumerate(path):
            self.step=k;a=raw[5*k:5*k+5];token=((a-mean)/std).reshape(10);outgoing=np.concatenate((history,token[None,None]),1)
            self.arrays('program_packet',latent=latent,history=history,raw=a,mean=mean,std=std,token=token,outgoing=outgoing)
            op=meter.begin('action_encoder');encoded=self.call('action_encoder',lambda:outgoing,{'0':outgoing},list(outgoing.shape));meter.end(op)
            predicted=np.asarray(c['latent_history'],np.float32)[None]
            op=meter.begin('predict')
            def native():
                body=self.call('predictor',lambda:predicted,{'0':latent,'1':encoded},list(latent.shape))
                return self.call('pred_proj',lambda:body.reshape(3,192),{'0':body.reshape(3,192)},[3,192]).reshape(1,3,192)
            y=self.call('predict',native,{'latent':latent,'act_emb':encoded},list(latent.shape));meter.end(op)
            latent=np.concatenate((latent[:,1:],y[:,-1:]),1);history=outgoing[:,1:]
            self.arrays('program_boundary',latent=latent,history=history)
            if kind=='native_packet':meter.emit(kind,packet=k,mode=mode,context=c)
            else:meter.emit(kind,packet=k,context=c)
        self.step=2
        self.journal.event(dict(kind='forward_finished',phase=self.phase,commands=raw.tolist(),contexts=path))
        return path
    def gradient(self,t,incumbent,meter):
        actions=np.asarray(incumbent['commands'],np.float32);path=self.forwards(t,actions,meter,'gradient_forward_packet')
        assert path==incumbent['contexts'];z=np.asarray(path[-1]['latent_history'][-1]);residual=z-np.asarray(t['goal'])
        before=dict(task_id=m.gradient.core.task_id(t),at_logits=incumbent['logits'],contexts=path,at_terminal=z.tolist(),
            native_float32_loss=float(np.mean(residual.astype(np.float32)**2)),saved_float64_cost=m.gradient.core.model_cost(t,z))
        meter.emit('gradient_forward_complete',forward=before)
        def backward():
            e=dict(index=int(meter.group.split('_')[-1])-1,resource='grad',evaluation_id=meter.group,completed=False,
                scalar_outputs=1,input_shape=[10,2],retain_graph=False,create_graph=False)
            meter.emit('AD_API_attempt',entry=e)
            std=np.tile(np.asarray(t['action_std'],np.float32),5).astype(float)
            j=np.column_stack(((.6*gradtest.F+.1*gradtest.G)/std,gradtest.F/std))
            g=(2*j.T@residual/192).reshape(10,2).astype(np.float32)
            def derivative():
                def callback():
                    self.arrays('raw_backward',len(self.ledger.events)-1,value=g);return g
                return self.ledger.call('raw_backward',callback)
            g=self.ledger.call('vjp',lambda:self.call('grad',derivative,{'loss':np.asarray(before['native_float32_loss'],np.float32),'raw':actions}),directions=1)
            meter.emit('AD_API_completed',entry=dict(e,completed=True));return g
        g=meter.call('backward',backward).astype(float);chain=1-np.asarray(incumbent['commands'],float)**2
        d=dict(**before,raw_control_gradient=g.tolist(),objective_gradient=(g*chain).reshape(2,10).tolist(),tanh_chain=chain.tolist(),scalar_reverse_sweeps=1)
        meter.emit('gradient_complete',gradient=d);return d


def runner(task,meter,ledger,journal):
    backend=RAMGradient(ledger,journal)
    with backend.guard(meter):return m.gradient.optimize(task,backend,meter)


@pytest.fixture(scope='module')
def complete():
    base,r,t=request();sink=Events()
    result=m.plan_node(r,t,None,None,sink,lambda:None,runner=runner,permit_ram=True)
    return base,r,t,result,sink.events


def test_full32_actions_only_and_complete_native_ledger(complete):
    _,r,t,result,events=complete;a=m.audit_plan(r,t,result,events)
    assert len(result['commands'])==5 and len(result['result']['final_prediction']['commands'])==10
    assert len(result['result']['candidates'])==33 and len(result['result']['gradients'])==32
    assert a['predict_API']==136 and a['scalar_grad']==a['VJP_directions']==32
    assert a['backward_engine_calls'] is a['FLOPs'] is None


def test_original_scalar_codec_promotes_transport_only(complete):
    scalar=np.asarray(1.,np.float32)
    assert scalar.shape==() and m.signal.unblob(m.signal.blob(scalar)).shape==(1,)
    assert m.signal.blob(scalar)==m.signal.blob(np.asarray([1.],np.float32))
    entries=[e['event']['entry'] for e in complete[-1] if e['kind']=='optimizer'
        and e['event']['kind'] in ('AD_API_attempt','AD_API_completed')]
    assert len(entries)==64 and all(e['scalar_outputs']==1 and e['input_shape']==[10,2] for e in entries)
    # The codec alone cannot distinguish a singleton vector from a scalar.
    # Rank certification is the source-bound runtime guard, not this blob.
    import inspect
    assert 'loss.ndim==0 and a.is_leaf and a.requires_grad and a.grad is None' in inspect.getsource(m.NativeGradient.gradient)


@pytest.mark.parametrize('fault',['long_blob','rawleaf_vector','gradient_vector'])
def test_scalar_rank_and_rawleaf_are_independently_strict(complete,fault):
    _,r,t,result,events=deepcopy(complete)
    kind,key={'long_blob':('io_input','grad:loss'),'rawleaf_vector':('io_input','grad:raw'),
        'gradient_vector':('io_return','grad:output')}[fault]
    e=next(e for e in events if e['kind']==kind and key in e['arrays'])
    a=m.signal.unblob(e['arrays'][key]);e['arrays'][key]=m.signal.blob(np.repeat(a,2) if fault=='long_blob' else a.reshape(-1))
    with pytest.raises(ValueError):m.audit_plan(r,t,result,events)


@pytest.mark.parametrize('field',['state','reward','success','env','goal7','future_rgb','mode','boundary','raw','older','current','Q','model'])
def test_actual_observation_firewall(field):
    base,r,t=request()
    if field in ('state','reward','success','env','goal7','future_rgb'):r[field]=[0]
    elif field=='mode':r['mode']='O'
    elif field=='boundary':r['boundary']=20
    elif field=='raw':r['raw10'][0][0]=.5
    elif field in ('older','current'):r['task']['root_context']['latent_history'][0 if field=='older' else -1][0]=1
    elif field=='Q':r['task']['projection'][0][0]=2
    else:r['task']['binding']['model_sha256']='d'*64
    with pytest.raises(ValueError):m.validate_request(r,base)


@pytest.mark.parametrize('fault',['unknown','missingIO','oplink','nativeout','raw','mean','grad','grad_loss','rawcallback','Q','winner','counter','dropcandidate','fullcommit','drop_reverse'])
def test_complete_native_buffers_and_math_failclosed(complete,fault):
    _,r,t,result,events=deepcopy(complete)
    if fault=='unknown':events.insert(1,dict(kind='native_task_success'))
    elif fault=='missingIO':events.remove(next(e for e in events if e['kind']=='io_input'))
    elif fault=='oplink':next(e for e in events if e['kind']=='io_input')['operation_index']=99999
    elif fault in ('raw','mean','nativeout','grad','grad_loss','rawcallback'):
        kind,key={'raw':('program_packet','raw'),'mean':('program_packet','mean'),'nativeout':('io_return','predict:output'),
            'grad':('io_return','grad:output'),'grad_loss':('io_input','grad:loss'),'rawcallback':('raw_backward','value')}[fault]
        e=next(e for e in events if e['kind']==kind and key in e['arrays']);a=m.signal.unblob(e['arrays'][key]).copy();a.flat[0]+=.1;e['arrays'][key]=m.signal.blob(a)
    elif fault=='Q':result['result']['candidates'][0]['cost']+=1
    elif fault=='winner':result['result']['winner_iteration']=99
    elif fault=='counter':result['resource_receipt']['completed']['predict_API']-=1
    elif fault=='dropcandidate':result['result']['candidates'].pop()
    elif fault=='fullcommit':result['commands']=result['result']['final_prediction']['commands']
    else:events.remove(next(e for e in events if e['kind']=='resource_attempt' and e['entry']['resource']=='grad'))
    with pytest.raises((ValueError,IndexError,KeyError)):m.audit_plan(r,t,result,events)


def test_no_global_AD_patch_and_original_scheduler_untouched():
    import ast
    source=Path(m.__file__).read_text();tree=ast.parse(source)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('install','setattr') for n in ast.walk(tree))
    assert 'GradGate(' not in source and m.gradient.UPDATES==32
    assert m.contract()['max_predict_API']==2176 and m.contract()['max_scalar_reverse_sweeps']==512


def test_AUTH_before_any_IO(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('must not read'))
    with pytest.raises(ValueError):m.run('/NONE','0'*64,'/NONE','wrong')


def test_exclusive_prepare_verify_and_drift(tmp_path,monkeypatch):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    material=dict(schema=m.NAME+'.prepare',RAM_only=True);monkeypatch.setattr(m,'build',lambda:deepcopy(material))
    path=tmp_path/'results'/(m.NAME+'_prepare_RAM.json');receipt=m.write(path,material)
    assert m.verify(path,receipt['sha256'])==material
    with pytest.raises(ValueError):m.write(path,material)
    path.write_bytes(m.raw(dict(material,RAM_only=False)))
    with pytest.raises(ValueError):m.verify(path,receipt['sha256'])


def test_actual_saved_known_templates_have_same_frozen_contract():
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as enc
    p=m.signal.read(m.ROOT/'results/agent_p3_same_feedback_physical20_20261028_v1_prepare_MAIN01.json')
    for i,layer in enumerate(p['layers']):
        t=enc.load_task(layer['task_export'],layer['task_sha256'])['gn_task']
        assert all(t[k]==p['tasks'][i][k] for k in ('root_context','goal','projection','action_mean','action_std'))
        assert t['packet_logit_cap']==float(.75*np.sqrt(5))


def execute_ram(*,fault=None,at=18,compile_failure=None,visual_failure=None,path='/RAM_GD'):
    h=p3test.physical_helpers;p,data=h.fixture();p['contract']=m.contract();p['tasks']=[];p['templates']=[]
    p['layers']=[next(x for x in p['layers'] if x['source_slot']==s and x['arm']==m.old.ARMS[0]) for s in (0,1)]
    data['visual_tasks']=data.pop('tasks')
    for s in (0,1):
        t=p3test.program_helpers.task(s);g=data['visual_tasks'][s]['gn_task']
        for k in ('root_context','goal','action_mean','action_std'):t[k]=deepcopy(g[k])
        t['binding'].update(context_sha256=m.digest(t['root_context']),goal_sha256=m.digest(t['goal']),normalizer_sha256=m.digest([t['action_mean'],t['action_std']]))
        p['tasks'].append(m.core.validate_task(t));p['templates'].append(deepcopy(g))
    store=p3test.RAM(path);envs=[];requests=[]
    def planner(request,template,model,device,journal,check,**kw):
        requests.append(deepcopy(request));b=request['boundary']
        if b==compile_failure:raise RuntimeError('RAM compile interrupted')
        commands=np.full((5,2),np.float32(.02+b*.002)).tolist()
        result=dict(status='sealed',commands=commands,resource_receipt=m.core.Ledger(m.node_caps()).receipt())
        journal.event(dict(kind='RAM_physical_flow_only',boundary=b));return result
    class Env(h.ArrayEnv):
        def __init__(self):super().__init__(fault=fault,at=at)
    def factory(meter):e=m.old.metered_class(Env,meter)();envs.append(e);return e
    encode=m.old.Handler.encode_observation
    def visual(self,request,sink,**runtime):
        if visual_failure is not None and f'_b{visual_failure:02}' in request['episode_id']:raise RuntimeError('RAM visual failure')
        return encode(self,request,sink,**runtime)
    with h.visual_mock(store) as (runtime,_),patch.object(m.p3,'NativeJournal',p3test.FullRamJournal),\
        patch.object(m,'plan_node',planner),patch.object(m,'sha',store.hash),patch.object(m.old,'sha',store.hash),\
        patch.object(m.old.Handler,'encode_observation',visual),patch.object(m.old.source,'sample_state',lambda *a:h.REQUEST.copy()):
        r=m.execute(p,data,store,m.old.Handler(None),runtime,factory,authorization=m.AUTH,
            final_check=lambda:dict(model_unchanged=True),permit_ram=True)
    store.test_p=p;store.test_data=data;return r,store,envs,requests


def test_actual_feedback_four_packets_all4_episode_slots():
    r,s,envs,requests=execute_ram();assert r['status']=='known_feedback_gradient_recorded',r['errors']
    assert len(r['rows'])==4 and len(r['nodes'])==len(requests)==16 and r['completed20']==4
    assert all(len(e.commands)==30 and e.closed for e in envs)
    assert r['counts']['encoder']==r['counts']['projector']==dict(attempted=20,completed=20)
    assert r['counts']['public_step']==dict(attempted=120,completed=120)
    assert m.physical_census(s.events,r)['complete']
    for request in requests:
        b=request['boundary'];row=next(x for x in r['rows'] if x['episode_id']==request['episode_id'])
        a=np.concatenate((m.old.kernel.moving.warmup_commands(),np.asarray(row['executed_raw'][:b],np.float32).reshape(-1,2)))[-10:]
        assert request['raw10']==a.tolist()
        if b==15:assert request['task']['root_context']['latent_history']==[[1.]*192]*3
        es=[e for e in s.events if e.get('episode_id')==row['episode_id']]
        encoded=next(e['index'] for e in es if e['kind']=='boundary_encoded' and e['actual_step']==b)
        sealed=next(e['index'] for e in es if e['kind']=='boundary_sealed' and e['boundary']==b)
        taken=[e for e in es if e['kind']=='authorized_dispatch' and e['packet_boundary']==b]
        assert encoded<sealed<taken[0]['index'] and len(taken)==5


@pytest.mark.parametrize('fault,at',[('terminate',18),('truncate',25),('throw',18)])
def test_native_stops_or_failure_keep_all4_and16(fault,at):
    # Original RAM environment uses 'terminal', not a new success rule.
    fault='terminal' if fault=='terminate' else fault
    r,s,envs,requests=execute_ram(fault=fault,at=at)
    assert len(r['rows'])==4 and len(r['nodes'])==16 and all(e.closed for e in envs)
    assert all(len(e.commands)<=30 for e in envs)
    assert all(n['status']=='not_reached' for n in r['nodes'] if n['boundary']>=(10 if at==18 else 15))


@pytest.mark.parametrize('failure',['compile','visual'])
def test_failed_feedback_node_no_stale_action_fallback(failure):
    kw={'compile_failure':10} if failure=='compile' else {'visual_failure':10}
    r,s,envs,requests=execute_ram(**kw)
    assert r['status']=='partial_or_failed' and len(r['rows'])==4 and len(r['nodes'])==16
    assert all(row['control_returned']==10 for row in r['rows'])
    assert all(n['status']=='failed' for n in r['nodes'] if n['boundary']==10)
    assert all(len(e.commands)==20 and e.closed for e in envs)


def test_physical_unknown_events_and_changed_caps_rejected():
    r,s,_,_=execute_ram();events=deepcopy(s.events)
    events.insert(1,dict(kind='unknown_physical_event'))
    with pytest.raises(ValueError):m.physical_census(events,r)
    r=deepcopy(r);r['contract']['physical_caps']['public_step']+=1
    with pytest.raises(ValueError):m.physical_census(s.events,r)
