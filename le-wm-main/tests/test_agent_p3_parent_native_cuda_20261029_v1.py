"""No model imports/production IO. Explicit numpy callback native-buffer fixtures."""
from copy import deepcopy
from pathlib import Path
import runpy
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as m

original=runpy.run_path(str(m.ROOT/'tests/test_agent_p3_parent_native_adapter_20261029_v1.py'))


class Memory:
    def __init__(self):self.events=[]
    def event(self,e):self.events.append(deepcopy(e))


class NumpyNative:
    kind='RAM_FIXTURE'
    def __init__(self,t,s,fail=False):self.task=t;self.sink=s;self.phase=None;self.step=None;self.ledger=None;self.fail=fail
    def arrays(self,kind,index=None,**values):
        self.sink.event(dict(kind=kind,scope=deepcopy(self.ledger.scope),phase=self.phase,step=self.step,
            operation_index=index,arrays={k:m.signal.blob(np.asarray(v,np.float32)) for k,v in values.items()}))
    def call(self,name,fn,inputs,shape=None):
        index=len(self.ledger.events)
        def actual():
            self.arrays('io_input',index,**{name+':'+k:v for k,v in inputs.items()});out=fn()
            self.arrays('io_return',index,**{name+':output':out});return out
        return self.ledger.call(name,actual,shape=shape)
    def trajectory(self,context,actions,program):
        state=np.asarray(context['latent_history'],np.float32)[None];history=np.asarray(context['action_history'],np.float32)[None];path=[]
        mean,std=(np.asarray(self.task[k],np.float32) for k in ('action_mean','action_std'))
        if not program:self.arrays('origin',latent=state,history=history,raw_flat=np.asarray(actions,np.float32).reshape(-1),mean=mean,std=std)
        for step,raw in enumerate(np.asarray(actions,np.float32).reshape(-1,5,2)):
            self.step=step;token=((raw-mean)/std).reshape(10);outgoing=np.concatenate((history,token[None,None]),axis=1)
            if program:self.arrays('program_packet',latent=state,history=history,raw=raw,mean=mean,std=std,token=token,outgoing=outgoing)
            else:self.arrays('packet',raw=raw,token=token,outgoing=outgoing)
            encoded=self.call('action_encoder',lambda:np.repeat(outgoing.mean(axis=2,keepdims=True),192,axis=2).astype(np.float32),{'0':outgoing},[1,3,10])
            def predict():
                def transition():
                    if self.fail:raise RuntimeError('RAM nested predictor failure')
                    z=state.copy();v=z[0,-1];clock=float(v[2])
                    v[0]+=np.float32(1+clock*1e8)*raw[:,0].sum(dtype=np.float32)
                    v[1]+=np.float32(clock>=5e-9)*raw[:,1].sum(dtype=np.float32);v[2]+=np.float32(1e-8)
                    return z
                z=self.call('predictor',transition,{'0':state,'1':encoded},[1,3,192])
                return self.call('pred_proj',lambda:z.reshape(3,192).copy(),{'0':z.reshape(3,192)},[3,192]).reshape(1,3,192)
            z=self.call('predict',predict,{'latent':state,'act_emb':encoded},[1,3,192])
            state=np.concatenate((state[:,1:],z[:,-1:]),axis=1);history=outgoing[:,1:]
            self.arrays('program_boundary' if program else 'boundary',latent=state,history=history)
            path.append(dict(context,latent_history=state[0].tolist(),action_history=history[0].tolist()))
        return path
    def replay(self,context,actions,segmented,ledger):
        self.ledger=ledger;self.phase='segmented' if segmented else 'whole'
        return self.trajectory(context,actions,True)
    def signal(self,req,ledger):
        self.ledger=ledger;self.step=None;h=req['horizon'];zero=np.zeros((5*h,2),np.float32)
        self.phase='nominal';n=np.asarray(self.trajectory(req['parent_context'],zero,False)[-1]['latent_history'][-1],np.float32);self.arrays('nominal',value=n)
        self.phase='AD'
        def jacobian():
            ad=np.asarray(self.trajectory(req['parent_context'],zero,False)[-1]['latent_history'][-1],np.float32);self.step=None;self.arrays('AD_terminal',value=ad)
            j=np.zeros((192,10*h),np.float32);clock=float(req['parent_context']['latent_history'][-1][2])
            for k in range(h):
                j[0,k*10:k*10+10:2]=np.float32(1+clock*1e8);j[1,k*10+1:k*10+10:2]=np.float32(clock>=5e-9);clock=float(np.float32(clock)+np.float32(1e-8))
            for i,row in enumerate(j):
                self.sink.event(dict(kind='direction',scope=deepcopy(ledger.scope),direction=i))
                basis=np.eye(192,dtype=np.float32)[i]
                def raw_return():self.arrays('raw_backward',len(ledger.events)-1,value=row);return row.copy()
                ledger.call('vjp',lambda:self.call('grad',lambda:ledger.call('raw_backward',raw_return),
                    {'output':ad,'raw':zero.reshape(-1),'basis':basis}),directions=1)
            return j
        j=ledger.call('jacobian',jacobian);self.arrays('jacobian',value=j)
        return dict(binding_sha256=req['binding_sha256'],parent_context_sha256=req['parent_context_sha256'],horizon=h,
            nominal=n.tolist(),jacobian=j.tolist(),producer_sha256=m.sha(m.signal.__file__))


def material():
    t=original['task']();u=original['task']();u['goal']=[1.]*192;u['binding']['goal_sha256']=m.digest(u['goal']);u['binding']['task_id']=m.digest('second fixed fixture identity')
    return dict(tasks=[t,u],profile=m.profile())


@pytest.fixture(scope='module')
def saved():
    p=material();store=Memory();rows=m.execute(p,lambda t,s:NumpyNative(t,s),store,permit_ram=True)
    return p,dict(rows=rows),store.events


def test_full_four_slot_native_buffers_original_tree_and_R(saved):
    p,r,events=saved
    assert [x['status'] for x in r['rows']]==['callback_progress_complete_NOT_certified']*3+['refused_unsupported_original_type']
    summaries=m.validate_saved(p,r,events,permit_fixture=True)
    assert len(summaries)==4
    assert summaries[0]['realized_progress_depth']==3 and summaries[1]['realized_progress_depth']==1
    assert r['rows'][3]['result']['queues'][0]['decision']['candidates']==8
    assert sum(x['status']=='refused_unsupported_original_type' for x in r['rows'][3]['result']['jobs'])==8
    assert all(not row['result'][k] for row in r['rows'] for k in m.parent.FLAGS)
    assert all(s['resources']['attempted']==s['resources']['completed'] for s in summaries)
    child=r['rows'][1]['result']['signal_mappings'][2]
    assert child['native_response']['jacobian']!=r['rows'][1]['result']['signal_mappings'][0]['native_response']['jacobian']


def first_call(events,method='signal'):
    first=next(e for e in events if e['kind']=='arm_event' and e['payload']['kind']=='native_call_attempt' and e['payload']['method']==method)
    return [deepcopy(e['payload']) for e in events if e['kind']=='arm_event' and e['slot']==first['slot'] and e['call']==first['call']]


@pytest.mark.parametrize('damage',('nominal','jacobian','raw','older','normalizer','grad','predictor','extra'))
def test_native_closed_buffer_mutations(saved,damage):
    _,_,events=saved;call=first_call(events)
    if damage=='extra':call.insert(-1,dict(kind='reward',value=1))
    elif damage=='normalizer':call[0]['task']['action_mean'][0]+=.25
    elif damage=='nominal':call[-1]['result']['nominal'][0]+=.25
    elif damage=='jacobian':call[-1]['result']['jacobian'][0][0]+=.25
    else:
        key,kind={'raw':('raw','packet'),'older':('latent','origin'),
            'grad':('grad:basis','io_input'),'predictor':('predictor:0','io_input')}[damage]
        event=next(x for x in call if x['kind']==kind and key in x['arrays'])
        a=m.signal.unblob(event['arrays'][key]).copy();a.flat[0]+=.25;event['arrays'][key]=m.signal.blob(a)
    with pytest.raises(ValueError):m.audit_native(call)


@pytest.mark.parametrize('damage',('path','action','context','phase'))
def test_replay_all_context_and_raw_mutations(saved,damage):
    call=first_call(saved[2],'replay')
    if damage=='path':call[-1]['result'][0]['latent_history'][0][0]+=.25
    elif damage=='action':call[0]['request']['actions'][0][0]+=.25
    elif damage=='context':call[0]['request']['parent_context']['action_history'][0][0]+=.25
    else:next(x for x in call if x['kind']=='program_packet')['phase']='nominal'
    with pytest.raises(ValueError):m.audit_native(call)


@pytest.mark.parametrize('kind',('reward','fake_native','resource_unknown'))
def test_unknown_event_refused(kind):
    with pytest.raises(ValueError):m.event_census([dict(kind='arm_event',slot=0,call=0,payload=dict(kind=kind))])


def test_runtime_failure_keeps_four_slots_and_nonrefundable_prefix():
    p=material();store=Memory();rows=m.execute(p,lambda t,s:NumpyNative(t,s,fail=True),store,permit_ram=True)
    assert rows[0]['status']=='failed_or_budget_exhausted' and all(r['status']=='blocked_after_runtime_failure' for r in rows[1:])
    result=rows[0]['result'];assert result['resource_receipt']['attempted']['predict_API']==1 and result['resource_receipt']['completed']['predict_API']==0
    summary=m.validate_saved(p,dict(rows=rows),store.events,permit_fixture=True)
    assert len(summary)==1 and not summary[0]['model_plan_available']


def test_whole_queue_budget_refusal_continues_other_slots():
    p=material();p['profile']['resource_caps']['fixed']['predict_API']=8
    store=Memory();rows=m.execute(p,lambda t,s:NumpyNative(t,s),store,permit_ram=True)
    assert rows[0]['status']==rows[2]['status']=='resource_cap_before_node_queue'
    assert rows[1]['result'] is not None and rows[3]['result'] is not None
    assert all(rows[i]['result']['resource_receipt']['attempted']['program_compile_calls']==0 for i in (0,2))
    assert len(m.validate_saved(p,dict(rows=rows),store.events,permit_fixture=True))==4


def test_recorded_callback_cannot_be_reused_for_another_parent(saved):
    call=first_call(saved[2]);state=dict(calls=[call],index=0);t=deepcopy(call[0]['task'])
    t['root_context']['latent_history'][0][0]=.25
    backend=m.SavedBackend(t,state)
    ledger=m.core.Ledger(m.parent.default_caps('adaptive'));ledger.scope=dict(test=True)
    with pytest.raises(ValueError):backend.signal(call[0]['request'],ledger)


def test_source_pin_no_old_edits_and_complete_caps():
    assert all(m.sha(p)==h for p,h in m.ADAPTER_FILES.items())
    assert m.profile()['resource_caps']['fixed']['predict_API']==114
    assert m.profile()['resource_caps']['adaptive']['predict_API']==562
    assert m.profile()['max_requests']==144 and len(m.roster())==4


def test_run_authorization_before_loading_or_verifying():
    with pytest.raises(ValueError,match='explicit MAIN'):m.run('absent','f'*64,'absent',authorization='no')


def test_cli_and_cold_no_neural_import():
    code=f'import sys; from experiments.spectral_reachability import {m.NAME} as m; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules; m.main(["--help"])'
    p=subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,capture_output=True,text=True)
    assert p.returncode==0 and all(x in p.stdout for x in ('prepare','verify','run','consume'))


def test_saved_production_never_accepts_RAM_implicitly(saved):
    p,r,events=saved;row=r['rows'][3];wrapped=[e for e in events if e['kind']=='arm_event' and e['slot']==3]
    with pytest.raises(ValueError,match='no implicit RAM'):m.validate_arm(p['tasks'][1],row,wrapped)


@pytest.mark.parametrize('damage',('missing_return','extra_io','charged_shape'))
def test_complete_resource_census_mutations(saved,damage):
    _,r,events=saved;row=r['rows'][3]
    body=[deepcopy(e['payload']) for e in events if e['kind']=='arm_event' and e['slot']==3]
    if damage=='missing_return':body.remove(next(x for x in body if x['kind']=='resource_return'))
    elif damage=='extra_io':body.append(deepcopy(next(x for x in body if x['kind']=='io_return')))
    else:
        e=next(x for x in body if x['kind']=='io_input' and 'action_encoder:0' in x['arrays'])
        e['arrays']['action_encoder:0']=m.signal.blob(np.zeros((1,3,20),np.float32))
    with pytest.raises(ValueError):m.resource_census(row['result']['resource_receipt'],body)


def test_original_decision_cannot_be_coherently_relabelled(saved):
    p,r,events=saved;row=deepcopy(r['rows'][3]);wrapped=[e for e in events if e['kind']=='arm_event' and e['slot']==3]
    row['result']['queues'][0]['decision']['extension_type']='progress'
    with pytest.raises(ValueError,match='saved choose'):m.validate_arm(p['tasks'][1],row,wrapped,permit_fixture=True)


def test_chained_saved_file_fixture_and_tampering(tmp_path,saved):
    events=[dict(kind='denominator',rows=[])];tail='0'*64;lines=[]
    for i,e in enumerate(events):
        x=dict(index=i,previous=tail,payload=e);tail=m.digest(x);x['sha256']=tail;lines.append(m.canonical(x)+b'\n')
    path=tmp_path/'events.jsonl';path.write_bytes(b''.join(lines))
    assert m.load_events(tmp_path,dict(journal_tail=tail))==events
    path.write_bytes(path.read_bytes().replace(b'denominator',b'unknown____'))
    with pytest.raises(ValueError):m.load_events(tmp_path,dict(journal_tail=tail))


def test_no_trailing_arm_work_after_return(saved):
    p,r,events=saved;copy=list(events);i=next(i for i,e in enumerate(copy) if e['kind']=='arm_event')
    event=copy.pop(i);copy.append(event)
    with pytest.raises(ValueError,match='inside real arm'):m.validate_saved(p,r,copy,permit_fixture=True)
