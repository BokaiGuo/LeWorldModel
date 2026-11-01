"""Explicit RAM native-array mocks; not real plans, models or physics evidence."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_actual_cp5_commit_physical_20261029_v1 as m
import test_agent_allmode_physical_20261028_v1 as old
from test_agent_p3_parent_native_adapter_20261029_v1 import task as model_task


def path_for(task,actions):
    """Only packet/history shifting, no predictor and no AD."""
    c=deepcopy(task['root_context']);out=[]
    for packet in np.asarray(actions,np.float32).reshape(-1,5,2):
        token=m.bridge.typed.tokens(packet,np.asarray(task['action_mean'],np.float32),np.asarray(task['action_std'],np.float32)).reshape(10)
        c=dict(c,latent_history=c['latent_history'][1:]+[c['latent_history'][-1]],action_history=[c['action_history'][-1],token.tolist()])
        out.append(deepcopy(c))
    return out


@lru_cache(maxsize=1)
def _fixture():
    prior,data,*_=old.fixture();reference=old.completed()[0];tasks=[];provenance=[];roots=[]
    for root in (0,1):
        row=reference['rows'][2*root];native=prior['source_roots'][root];prefix=np.asarray(row['plan']['commands'][:5],np.float32)
        warm=m.moving.warmup_commands();raw=np.concatenate((warm[-5:],prefix));base=model_task()
        base['binding'].update(root_sha256=native['root_sha256'],task_id=row['task_id'])
        local=deepcopy(base);local['binding']['task_id']=m.producer_digest(['RAM actual cp5',root])
        local['root_context']['action_history']=m.bridge.typed.tokens(raw,np.asarray(local['action_mean'],np.float32),np.asarray(local['action_std'],np.float32)).reshape(2,10).tolist()
        local['binding']['context_sha256']=m.producer_digest(local['root_context'])
        pro=dict(original_task=deepcopy(base),original_task_sha256=m.producer_digest(base),local_task_sha256=m.producer_digest(local),
            actual_cp5_absolute_steps=[5,10,15],actual_cp5_control_steps=[-5,0,5],prefix5=prefix.tolist(),raw10=raw.tolist())
        frames=[]
        for phase,step,absolute in (('warmup',5,5),('warmup',10,10),('plan',5,15)):
            f=next(x['receipt'] for x in row['prefix51']['frames'] if (x['receipt']['phase'],x['receipt']['step'],x['receipt']['overlay_index'])==(phase,step,0))
            frames.append(dict(absolute_step=absolute,plan_step=absolute-10,**{k:deepcopy(f[k]) for k in ('png','file_sha256','pixel_sha256','state','snapshot')}))
        cp=deepcopy(next(c for c in row['checkpoints'] if c['step']==5));cp['absolute_step']=15;cp['plan_step']=5
        anchor=dict(prefix_raw5=prefix.tolist(),warm_raw10=warm.tolist(),checkpoint=cp,prefix51=deepcopy(row['prefix51']['frames']),
            condition='off',root_sha256=native['root_sha256'],goal=native['goal'],goal_pose=native['goal_pose'])
        roots.append(dict(actual_cp5_task=deepcopy(local),candidate_base_task=deepcopy(base),anchor=anchor,frames=frames,
            history=dict(candidate_context=deepcopy(local['root_context']),all_three_frames_actual_verified=True,raw10=raw.tolist(),history_tokens=deepcopy(local['root_context']['action_history']))))
        tasks.append(local);provenance.append(pro)
    args={k:('/RAM/'+k if not k.endswith('_sha') else old.h(k)) for k in m.ARGUMENTS}
    c=dict(schema=m.PRODUCER+'.prepare',tasks=tasks,provenance=provenance,rules=dict(protocol_sha256=m.PROTOCOL_SHA),sources={})
    arms=[]
    for i,(root,h) in enumerate(m.ORDER):
        task=tasks[root];pro=provenance[root];edge=np.full((5*h,2),np.float32(.125)).tolist();path=path_for(task,edge)
        ver=[dict(mode=mode,status='returned',path=deepcopy(path),segments=[5*h] if mode=='whole' else ([5] if h==1 else [5,10])) for mode in ('whole','segmented')]
        d=dict(schema=m.PRODUCER+'.delivery',model_delivery_verified=True,task=deepcopy(task),base_task=deepcopy(pro['original_task']),
            source_closure_sha256=m.producer_digest(c['sources']),root_context_role='actual_observed_abs5_10_15_control_minus5_0_5',
            task_sha256=m.producer_digest(task),base_task_sha256=m.producer_digest(pro['original_task']),provenance_sha256=m.producer_digest(pro),
            raw_edge=edge,commit_raw5=deepcopy(edge[:5]),lookahead_raw=deepcopy(edge[5:]),raw_edge_float32_sha256=m.bytehash(edge),
            commit_float32_sha256=m.bytehash(edge[:5]),lookahead_float32_sha256=m.bytehash(np.asarray(edge[5:],np.float32).reshape(-1,2)),
            control_start=5,commit_control_steps=[6,7,8,9,10],prediction_end_control=5+5*h,prediction_end_absolute=15+5*h,
            suffix_in_plan=False,verification=ver,path=path,physical_release=False,execution_authorized=False,P1_admission=False,P3_admission=False,C3=False)
        ledger=m.bridge.typed.core.Ledger(dict.fromkeys(m.bridge.typed.core.AXES,10));ledger.scope={'fixture':True};ledger.call('replay',lambda:None)
        arms.append(dict(slot=i,root=root,horizon=h,complete=True,status='model_delivery_verified',model_delivery_verified=True,no_witness=False,delivery=d,
            resources=ledger.receipt(),scalar_backward_calls=None,total_FLOPs=None))
    u=dict(schema=m.PRODUCER+'.consume',status='actual_cp5_R_model_deliveries_verified_NOT_physical',report_sha256=args['delivery_report_sha'],
        prepare_sha256=args['delivery_prepare_sha'],prepared=args['delivery_prepared'],denominator=4,protocol_sha256=m.PROTOCOL_SHA,arms=arms,
        physical_release=False,execution_authorized=False,P1_admission=False,P3_admission=False,C3=False)
    b=dict(schema=m.bridge.NAME+'.bridge',status='BLOCKED',all_three_frames_actual_verified=True,roots=roots,
        typed_contract_partial=dict(authorities=dict(planner=None)))
    return args,c,u,b,data


def fixture():
    args,c,u,b,data=deepcopy(_fixture());p=m.assemble(args,c,u,b,data);return p,data,args,c,u,b


def test_real_delivery_abi_four_slots_and_distinct_codecs():
    p,*_=fixture();m.validate_execution(p)
    assert m.digest([1,2])!=m.producer_digest([1,2])
    assert [(p['root_slot'],p['horizon']) for p in p['plans']]==m.ORDER
    assert all(x['task_id']!=x['local_task_id'] for x in p['plans'])
    assert all(len(x['commands'])==10 for x in p['plans'])
    assert [len(x['model_delivery']['lookahead_raw']) for x in p['plans']]==[0,10,0,10]
    assert all(p['planner_authority'][k] is False for k in m.FLAGS)


@pytest.mark.parametrize('kind',['schema','partial','order','sha','base_task','local_task','raw','commit','lookahead','whole','segments','flags','history','goal','prefix','time'])
def test_real_delivery_and_history_mutations_fail_closed(kind):
    _,data,args,c,u,b=fixture();d=u['arms'][1]['delivery']
    if kind=='schema':u['schema']='oldplanner.consume'
    elif kind=='partial':u['arms'][0]['complete']=False
    elif kind=='order':u['arms'].reverse()
    elif kind=='sha':u['report_sha256']=old.h('wrong')
    elif kind=='base_task':d['base_task_sha256']=m.digest(d['base_task'])
    elif kind=='local_task':d['task_sha256']=old.h('wrong')
    elif kind=='raw':d['raw_edge'][0][0]=.25
    elif kind=='commit':d['commit_raw5'][0][0]=.25
    elif kind=='lookahead':d['lookahead_raw'][0][0]=.25
    elif kind=='whole':d['verification'][0]['path']=[]
    elif kind=='segments':d['verification'][1]['segments']=[15]
    elif kind=='flags':d['physical_release']=True
    elif kind=='history':b['roots'][0]['history']['candidate_context']['latent_history'][0][0]=.25
    elif kind=='goal':b['roots'][0]['anchor']['goal'][0]+=.25
    elif kind=='prefix':b['roots'][0]['anchor']['prefix_raw5'][0][0]=.25
    else:c['provenance'][0]['actual_cp5_absolute_steps']=[0,5,10]
    with pytest.raises((ValueError,KeyError)):m.assemble(args,c,u,b,data)


def execute(p=None,data=None,*,fault=None,at=18):
    if p is None:p,data,*_=fixture()
    store=old.MemoryStore();envs=[]
    def factory():e=old.Env(fault=fault,at=at);envs.append(e);return e
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m.native,'place',old.fake_place);mp.setattr(m.native,'validate_env',lambda _:None)
        mp.setattr(m,'sha',lambda f:hashlib.sha256(store.files[Path(f).name]).hexdigest())
        r=m.execute(p,data,store,factory,authorization=m.AUTH,prepared='/RAM/own_prepare.json',prepare_sha256=old.h('own prepare'))
    return r,store,p,data,envs


@lru_cache(maxsize=1)
def complete():return execute()


def test_full_lifecycle_native_array_callbacks_and_saved_validator():
    r,s,p,data,envs=complete();assert r['status']==m.RECORDED,r['errors']
    assert len(m.validate_saved(r,s.events,p))==4 and len(envs)==4 and all(e.closed for e in envs)
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()}
    assert len([x for x in s.files if x.endswith('.png')])==264
    for i,(row,env) in enumerate(zip(r['rows'],envs,strict=True)):
        assert len(env.commands)==20 and row['own_returned']==10
        assert np.asarray(env.commands[10:],np.float32).tobytes()==np.asarray(p['plans'][i]['commands'],np.float32).tobytes()
        guard=next(x for x in s.events if x['kind']=='cp5_guard' and x['branch']==i)
        step6=next(x for x in s.events if x['kind']=='operation_dispatch_intent' and x['receipt']['branch']==i
            and x['receipt']['resource']=='public_step' and x['receipt'].get('phase')=='plan' and x['receipt'].get('step')==6)
        assert guard['accepted'] and guard['index']<step6['index']


def reseal_plan(p,i):
    p['plans'][i]['prefix_reference_sha256']=m.digest(p['prefix_reference'][i])
    p['plans'][i]['selection_digest']=m.digest({k:v for k,v in p['plans'][i].items() if k!='selection_digest'})
    p['planner_authority']['plans_sha256']=m.digest(p['plans'])


@pytest.mark.parametrize('kind',['state','snapshot','pixels','png','warmraw','history_frame','tokens'])
def test_cp5_guard_mismatch_never_dispatches_any_commit(kind):
    p,data,*_=fixture()
    for i,ref in enumerate(p['prefix_reference']):
        if kind=='state':ref['checkpoint']['state'][0]+=.25
        elif kind=='snapshot':ref['checkpoint']['snapshot']['state_float64_bits']='wrong'
        elif kind=='pixels':ref['checkpoint']['pixel_sha256']=old.h('wrong')
        elif kind=='png':ref['frames'][-3]['receipt']['file_sha256']=old.h('wrong')
        elif kind=='warmraw':ref['warm_raw10'][0][0]=.25
        elif kind=='history_frame':ref['history_frames'][0]['pixel_sha256']=old.h('wrong')
        else:ref['history_tokens'][0][0]+=.25
        reseal_plan(p,i)
    r,s,_,_,envs=execute(p,data)
    assert r['status']=='partial_or_failed' and r['errors']
    assert envs and len(envs[0].commands)==15
    assert not any(e['kind']=='operation_dispatch_intent' and e['receipt']['resource']=='public_step' and
        e['receipt'].get('phase')=='plan' and e['receipt'].get('step',0)>=6 for e in s.events)
    assert len(m.validate_saved(r,s.events,p))==4


def test_no_witness_all4_is_zero_callbacks_retains_nulls():
    _,data,args,c,u,b=fixture()
    for arm in u['arms']:arm.update(status='no_witness',model_delivery_verified=False,no_witness=True,delivery=None)
    p=m.assemble(args,c,u,b,data);r,s,_,_,envs=execute(p,data)
    assert r['status']==m.RECORDED and not envs and not r['actual_operation_receipts']
    assert all(row['final_state'] is None and row['missing_checkpoints']==[0,5,10] for row in r['rows'])
    assert len(m.validate_saved(r,s.events,p))==4


def test_mixed_witness_retains_four_slots_and_actual_resources_only():
    _,data,args,c,u,b=fixture()
    for i in (0,3):u['arms'][i].update(status='no_witness',model_delivery_verified=False,no_witness=True,delivery=None)
    p=m.assemble(args,c,u,b,data);r,s,_,_,envs=execute(p,data)
    assert r['status']==m.RECORDED and len(envs)==2 and len(m.validate_saved(r,s.events,p))==4
    assert r['counts']['public_step']['completed']==40 and r['counts']['reset']['completed']==4
    assert [x['status'] for x in r['rows']]==['no_witness','horizon_completed','horizon_completed','no_witness']


@pytest.mark.parametrize('at',[5,14,18])
def test_truncation_keeps_checkpoint_missing_and_denominator(at):
    r,s,p,_,envs=execute(fault='truncate',at=at)
    assert r['status']==m.RECORDED,r['errors']
    assert len(m.validate_saved(r,s.events,p))==4 and all(len(e.commands)==at for e in envs)
    assert all(10 in row['missing_checkpoints'] and row['status']=='native_stopped' for row in r['rows'])


def reseal(events,r):old.reseal(events,r)


@pytest.mark.parametrize('kind',['unknown','guard_missing','guard_late','guard_proof','guard_false','action','label','counter','missing_frame'])
def test_saved_trace_tampering_rejected(kind):
    r,s,p,*_=deepcopy(complete());events=s.events
    if kind=='unknown':events[1]['kind']='caller_says_valid'
    elif kind=='guard_missing':events[:]=[e for e in events if e['kind']!='cp5_guard']
    elif kind=='guard_late':
        i=next(i for i,e in enumerate(events) if e['kind']=='cp5_guard');g=events.pop(i);events.insert(i+2,g)
    elif kind=='guard_proof':next(e for e in events if e['kind']=='cp5_guard')['proof']['actual_raw10_float32_sha256']=old.h('wrong')
    elif kind=='guard_false':next(e for e in events if e['kind']=='cp5_guard')['accepted']=False
    elif kind=='action':next(e for e in events if e['kind']=='returned_transition' and e['phase']=='plan')['actual_bits']=np.asarray([.5,.5],np.float32).tobytes().hex()
    elif kind=='label':next(e for e in events if e['kind']=='returned_transition')['reward']+=1
    elif kind=='counter':r['counts']['physics_step']['completed']-=1
    else:events.pop(next(i for i,e in enumerate(events) if e['kind']=='fresh_observation'))
    reseal(events,r)
    with pytest.raises((ValueError,KeyError)):m.validate_saved(r,events,p)


def test_guard_journal_failure_never_enters_native_callback():
    r,s,p,*_=deepcopy(complete());row=r['rows'][0];ref=p['prefix_reference'][0];plan=p['plans'][0]
    class Broken:
        def emit(self,*a,**kw):raise OSError('RAM disk fault at guard fsync')
    meter=m.Meter(Broken());meter.where=dict(branch=0);meter.guard_context(row,ref,plan)
    meter.receipts=[x for x in r['actual_operation_receipts'] if x['branch']==0 and x['resource']=='public_step'
        and (x['phase']=='warmup' or x['step']<=5)]
    called=[]
    with pytest.raises(OSError):meter.call('public_step',lambda:called.append(True),phase='plan',step=6)
    assert not called and not meter.guarded


def test_actual_store_roundtrip_full_mock_lifecycle_and_png_consumer(tmp_path,monkeypatch):
    p,data,*_=fixture();prepare=tmp_path/'mock_prepare.json';prepare.write_bytes(m.raw(p));expected=m.sha(prepare)
    out=tmp_path/(m.NAME+'_RAM');store=m.Store(out);envs=[]
    def factory():e=old.Env();envs.append(e);return e
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m.native,'place',old.fake_place);mp.setattr(m.native,'validate_env',lambda _:None)
        r=m.execute(p,data,store,factory,authorization=m.AUTH,prepared=str(prepare),prepare_sha256=expected)
    assert r['status']==m.RECORDED,r['errors']
    # Only upstream preparation is a RAM fixture seam. Real disk writer, journal,
    # native callback hooks, PNG decoding and entire own saved consumer execute.
    monkeypatch.setattr(m,'verify',lambda path,h:(deepcopy(p),deepcopy(data)))
    result=m.consume(out,m.sha(out/'report.json'))
    assert result['saved_PNG_verified']==264 and len(result['rows'])==4
    assert all(result[k] is False for k in m.FLAGS)
    with pytest.raises(ValueError):m.Store(out)
    png=next(out.glob('*.png'));body=png.read_bytes();png.write_bytes(body+b'corrupt')
    with pytest.raises(ValueError):m.consume(out,m.sha(out/'report.json'))


def test_real_store_two_events_hashchain_reader(tmp_path):
    store=m.Store(tmp_path/(m.NAME+'_writer_fixture'))
    store.emit('RAM_one',value=1);store.emit('RAM_two',value=2)
    events=[m.native.decode(x) for x in (store.path/'journal.jsonl').read_bytes().splitlines()]
    assert events[0]['index']==0 and events[1]['index']==1
    assert events[1]['previous_sha256']==events[0]['event_sha256']==m.source.digest({k:v for k,v in events[0].items() if k!='event_sha256'})
    assert events[1]['event_sha256']==store.previous


def test_authorization_before_factory():
    p,data,*_=fixture();called=[]
    with pytest.raises(ValueError):m.execute(p,data,old.MemoryStore(),lambda:called.append(True),authorization='wrong',
        prepared='/RAM/p.json',prepare_sha256=old.h('p'))
    assert not called


def test_import_no_neural_physics_and_no_delivery_producer():
    code=f'from experiments.spectral_reachability import {m.NAME}; import sys; assert "torch" not in sys.modules and "pymunk" not in sys.modules; assert not any("actual_cp5_r_delivery" in k for k in sys.modules)'
    result=subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
