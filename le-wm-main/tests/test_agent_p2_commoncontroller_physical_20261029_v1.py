"""RAM/fixture and selected saved ABI only; no env construction or NN production."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_commoncontroller_physical_20261029_v1 as m
from test_agent_p2_three_program_physical_20261029_v1 import fixture as kernel_fixture
from test_agent_allmode_physical_20261028_v1 import MemoryStore,Env,fake_place,h,reseal


RECEIPT=m.ROOT/'results/agent_p2_delivered_mean_paid_replay_cuda_20261029_v1_consume_MAIN01.json'
RECEIPT_SHA='a8fe4dd9c8c90ed14a5bbecdcb226ec4d7626c786fe4db1223194f7e3f6d87b1'


def receipt_shape():
    assert m.sha(RECEIPT)==RECEIPT_SHA
    return json.loads(RECEIPT.read_bytes())


def args():
    return {k:(h(k) if k.endswith('_sha') else '/RAM/'+k+'.json') for k in m.ARGS}


def fixture():
    original,data,_,_=kernel_fixture();a=args();receipt=receipt_shape()
    receipt.update(manifest_sha256=a['replay_manifest_sha'],report_sha256=a['replay_report_sha'])
    mean=np.asarray([-.0087,.0068],np.float32);std=np.asarray([.2019,.2002],np.float32)
    past=np.divide(np.subtract(m.moving.warmup_commands(),mean,dtype=np.float32),std,dtype=np.float32).reshape(1,2,10)
    manifest=dict(schema=m.PAID+'.prepare',controller_prepare=dict(delivery_artifact=dict(
        rows=[],denominator=dict(roots=2,arms=3,total=6,assembled=4,missing=2,dispatched=0))),raw_artifact=dict(rows=[]))
    report=dict(schema=m.PAID+'.report',completed=True,errors=[],manifest_sha256=a['replay_manifest_sha'],rows=[])
    arrays={};descs={}
    for i,(root,condition,arm) in enumerate(m.ORDER):
        s=original['plans'][2*root+1]['planner_plan']
        d=dict(root=root,condition=condition,arm=arm,**{k:s[k] for k in ('task_id','task_sha256','blind_sha256')},
            status='missing',controller_payload=None)
        r=dict(slot=i,root=root,arm=arm,status='missing')
        rd=dict(checks=None)
        if i in m.ACTIVE:
            z=np.asarray(s['root_context']['latent_history'],np.float32)[None]
            s['root_context']['action_history']=past[0].tolist()
            target=np.full((1,192),np.float32(.125+i),np.float32)
            raw=np.full((15,2),np.float32(.125*(i%3)),np.float32)
            d.update(status='assembled_not_dispatched',controller_payload=dict(z=z.tolist(),a=past.tolist(),target=target.tolist()),
                origin=dict(role='RAM_only'))
            r['status']='returned';arrays[i]=(raw,z,past.copy(),target,mean.copy(),std.copy())
            descs[i]=dict(file=f'slot{i}_raw.npy',file_sha256=h(('raw',i)),root=root,shape=[15,2],dtype='<f4',data_sha256=m.byte_sha(raw),file_bytes=248)
            rd=dict(checks=dict(array_bits=dict(raw=raw.tobytes().hex())))
        manifest['controller_prepare']['delivery_artifact']['rows'].append(d)
        manifest['raw_artifact']['rows'].append(rd);report['rows'].append(r)
    p=m.assemble(a,manifest,report,receipt,original,data,arrays,descs)
    return p,data,(a,manifest,report,receipt,original,arrays,descs)


def test_actual_receipt_shape_and_protocol_pin():
    r=receipt_shape();a=args();a.update(replay_manifest_sha=r['manifest_sha256'],replay_report_sha=r['report_sha256'])
    m.paid_complete(r,a)
    assert 'schema' not in r and 'physical_release' not in r
    assert m.sha(m.PROTOCOL)==m.PROTOCOL_SHA
    assert m.digest([1,2])!=hashlib.sha256(b'[1,2]').hexdigest()


def test_six_slot_own_plan_and_fixed_physical_caps():
    p,_,_=fixture();m.validate_execution(p)
    assert [x['status'] for x in p['plans']]==['sealed_raw15','sealed_raw15','missing','sealed_raw15','sealed_raw15','missing']
    assert m.contract()['caps']==m.kernel.contract()['caps']
    assert m.contract()['caps']['physics_step']==1016
    assert m.contract()['fixed_denominator']==6
    assert m.contract()['arm']=='commoncontroller_returned_mean_raw15'


@pytest.mark.parametrize('field',['flag','flagmissing','status','subset','blocked','predict','tokens','order','manifest','report'])
def test_paid_actual_shape_mutations_failclosed(field):
    p,_,_=fixture();a=p['arguments'];r=deepcopy(p['planner_authority']['receipt'])
    if field=='flag':r['raw_deployment_authorized']=True
    elif field=='flagmissing':del r['raw_deployment_authorized']
    elif field=='status':r['status']='ready'
    elif field=='subset':r['denominator']['returned']=3
    elif field=='blocked':r['local_checks'][0]['blocked']='blocked'
    elif field=='predict':r['predict_API']=23
    elif field=='tokens':r['module_resources']['AE_tokens']=95
    elif field=='order':r['local_checks'].reverse()
    elif field=='manifest':r['manifest_sha256']=h('bad')
    else:r['report_sha256']=h('bad')
    with pytest.raises((ValueError,KeyError)):m.paid_complete(r,a)


@pytest.mark.parametrize('field',['raw','rawzero','task','context','history','target','order','missing','mean'])
def test_assemble_identity_mutations(field):
    _,data,values=fixture();a,man,r,rc,old,arr,descs=deepcopy(values)
    if field=='raw':arr[0][0][14,1]=.25
    elif field=='rawzero':arr[0][0][0,0]=-0.
    elif field=='task':man['controller_prepare']['delivery_artifact']['rows'][0]['task_id']=h('bad')
    elif field=='context':arr[0][1][0,0,0]+=.1
    elif field=='history':arr[0][2][0,0,0]+=.1
    elif field=='target':arr[0][3][0,0]+=.1
    elif field=='order':r['rows'].reverse()
    elif field=='missing':man['controller_prepare']['delivery_artifact']['rows'][2]['controller_payload']={}
    else:arr[0][4][0]+=.01
    with pytest.raises(ValueError):m.assemble(a,man,r,rc,old,data,arr,descs)


def execute(fault=None,at=11,close_error=False,corrupt=None,guard_write_failure=False):
    p,data,_=fixture();s=MemoryStore();envs=[]
    def factory():
        e=Env(fault=fault,at=at,close_error=close_error);envs.append(e);return e
    proof=m.initial_proof
    def check(plan,prepared,events):
        if corrupt:
            events=deepcopy(events)
            if corrupt=='state':events[-1]['receipt']['state'][0]+=.01
            elif corrupt=='pixels':events[-1]['receipt']['pixel_sha256']=h('bad')
            elif corrupt=='raw':next(x for x in events if x['kind']=='returned_transition')['actual_bits']='00'
            elif corrupt=='goal':next(x for x in events if x['kind']=='goal_attached')['goal'][0]+=.01
            else:events.pop()
        return proof(plan,prepared,events)
    emit=s.emit
    def fail_emit(kind,**fields):
        if guard_write_failure and kind=='initial_history_guard':raise OSError('RAM guard fsync failure')
        return emit(kind,**fields)
    s.emit=fail_emit
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m,'place',fake_place);mp.setattr(m,'validate_env',lambda _:None)
        mp.setattr(m,'initial_proof',check)
        mp.setattr(m,'sha',lambda path:hashlib.sha256(s.files[Path(path).name]).hexdigest())
        r=m.execute(p,data,s,factory,authorization=m.AUTH,prepared='/RAM/own.json',prepare_sha256=h('own'))
    return r,s,p,data,envs


@lru_cache(maxsize=1)
def complete():return execute()


def test_complete_real_loop_ram_saved_consumer_324png_six_denominator():
    r,s,p,_,envs=complete();assert r['status']==m.RECORDED,r['errors']
    assert len(m.validate_saved(r,s.events,p))==6
    assert len(envs)==4 and all(e.closed and len(e.commands)==25 for e in envs)
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()}
    assert len([n for n in s.files if n.endswith('.png')])==324
    for i,e in zip(m.ACTIVE,envs,strict=True):
        row=r['rows'][i];assert row['missing_checkpoints']==[]
        assert np.asarray(e.commands[10:],np.float32).tobytes()==m.raw15(p['plans'][i]['commands']).tobytes()
        guard=next(x for x in s.events if x['kind']=='initial_history_guard' and x['branch']==i)
        intent=next(x for x in s.events if x['kind']=='operation_dispatch_intent' and x['receipt']['branch']==i
            and x['receipt']['resource']=='public_step' and x['receipt'].get('phase')=='plan')
        assert guard['index']<intent['index']
    for i in (2,5):assert r['rows'][i]==m.missing_row(p['plans'][i])


@pytest.mark.parametrize('corrupt',['state','pixels','raw','goal','missing'])
def test_guard_mismatch_absolutely_no_plan_intent_or_callback(corrupt):
    r,s,p,_,envs=execute(corrupt=corrupt)
    assert r['status']=='partial_or_failed'
    assert all(len(e.commands)==10 and e.closed for e in envs)
    assert not any(x['kind']=='operation_dispatch_intent' and x['receipt'].get('phase')=='plan' for x in s.events)
    assert not any(x['kind']=='initial_history_guard' for x in s.events)
    assert all(r['rows'][i]['own_attempted']==0 for i in m.ACTIVE)


def test_guard_durable_write_failure_absolutely_no_plan_callback():
    r,s,_,_,envs=execute(guard_write_failure=True)
    assert all(len(e.commands)==10 for e in envs) and r['status']=='partial_or_failed'
    assert not any(x['kind']=='operation_dispatch_intent' and x['receipt'].get('phase')=='plan' for x in s.events)


@pytest.mark.parametrize('fault,at',[('truncate',5),('truncate',11),('terminal',25),('throw',11)])
def test_native_stops_or_failed_calls_preserve_denominator(fault,at):
    r,s,p,_,envs=execute(fault=fault,at=at)
    assert len(m.validate_saved(r,s.events,p))==6
    assert all(e.closed and len(e.commands)==at for e in envs)
    assert all((15 in r['rows'][i]['missing_checkpoints'])==(at<25) for i in m.ACTIVE)
    assert r['status']==('partial_or_failed' if fault=='throw' else m.RECORDED)


@pytest.mark.parametrize('field',['unknown','late_guard','guard_missing','guard_payload','cp15','missing','reward','raw','count','binding','prefix'])
def test_saved_tampering_even_resealed(field):
    r0,s,p0,_,_=complete();r,p,ev=deepcopy(r0),deepcopy(p0),deepcopy(s.events)
    if field=='unknown':ev.insert(-1,dict(kind='unknown'))
    elif field=='late_guard':
        g=next(x for x in ev if x['kind']=='initial_history_guard');ev.remove(g);ev.insert(-2,g)
    elif field=='guard_missing':ev.remove(next(x for x in ev if x['kind']=='initial_history_guard'))
    elif field=='guard_payload':next(x for x in ev if x['kind']=='initial_history_guard')['proof']['raw_float32_sha256']=h('bad')
    elif field=='cp15':r['rows'][0]['checkpoints'].pop()
    elif field=='missing':r['rows'][2]['native_success']=False
    elif field=='reward':next(x for x in ev if x['kind']=='returned_transition')['reward']=0.
    elif field=='raw':next(x for x in ev if x['kind']=='operation_dispatch_intent' and x['receipt']['resource']=='public_step')['receipt']['raw_bits']='00'
    elif field=='count':r['counts']['reset']['completed']-=1
    elif field=='binding':next(x for x in ev if x['kind']=='goal_attached')['goal'][0]+=.01
    else:next(x for x in ev if x['kind']=='fresh_observation')['receipt']['pixel_sha256']=h('bad')
    reseal(ev,r)
    with pytest.raises((ValueError,KeyError,StopIteration)):m.validate_saved(r,ev,p)


def test_full_png_consume_from_ram_loop(monkeypatch):
    r,s,p,data,_=complete()
    monkeypatch.setattr(m,'saved_artifacts',lambda _:deepcopy(s.files))
    monkeypatch.setattr(m,'verify',lambda *a:(deepcopy(p),data))
    result=m.consume('/RAM/physical',hashlib.sha256(s.files['report.json']).hexdigest())
    assert result['saved_PNG_verified']==324 and result['fixed_denominator']==6
    assert result['metric_breakdown'][2]['metrics'] is None
    monkeypatch.setattr(m,'saved_artifacts',lambda _:dict(s.files,extra=b'bad'))
    with pytest.raises(ValueError):m.consume('/RAM/physical',hashlib.sha256(s.files['report.json']).hexdigest())


def test_no_auth_no_reads_no_callbacks(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('read'))
    with pytest.raises(ValueError):m.run('/RAM/prep',h('prep'),'/RAM/output',lambda:pytest.fail('env'),authorization='bad')


def test_cold_cli_no_env_or_neural_import():
    code="import sys;from experiments.spectral_reachability import agent_p2_commoncontroller_physical_20261029_v1;assert not any(k in sys.modules for k in ('torch','gymnasium','pymunk','stable_worldmodel.envs'))"
    subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,check=True,capture_output=True)


def test_actual_saved_abi_selected_arrays_assemble_only(monkeypatch):
    """Read saved actual selected arrays, NEVER producer.verify/consume/source_material."""
    from experiments.spectral_reachability import agent_p2_delivered_mean_paid_replay_cuda_20261029_v1 as paid
    rp=m.ROOT/'results/agent_p2_delivered_mean_paid_replay_cuda_20261029_v1_cuda_MAIN01/report.json'
    report=json.loads(rp.read_bytes());receipt=receipt_shape()
    assert m.sha(rp)==receipt['report_sha256']
    mp=Path(report['manifest']);assert m.sha(mp)==receipt['manifest_sha256']
    manifest=json.loads(mp.read_bytes());original=json.loads(m.ORIGINAL.read_bytes())
    assert m.sha(m.ORIGINAL)==m.ORIGINAL_SHA
    a=dict(replay_manifest=str(mp),replay_manifest_sha=receipt['manifest_sha256'],
        replay_report=str(rp),replay_report_sha=receipt['report_sha256'],
        replay_receipt=str(RECEIPT),replay_receipt_sha=RECEIPT_SHA)
    arrays={};descs={};roots=[]
    for root in (0,1):
        plan=original['plans'][2*root+1]['planner_plan']
        # Original NPZ payload only; no native construction or upstream strict.
        packet=m.moving.load_blind(plan['blind_path'],plan['blind_sha256'])
        roots.append(dict(slot=original['source_slots'][root],warm_trace=original['warm_traces'][root],
            packets=dict(off=packet,on=packet)))
    data=dict(roots=roots)
    for i in m.ACTIVE:
        row=report['rows'][i];folder=rp.parent
        trace=paid.read(folder/row['trace']['file'],row['trace']['file_sha256'])
        descs[i]=row['arrays'][trace['arithmetic'][1]['output']]
        load=lambda d:paid.controller.load_array(folder,d)
        arrays[i]=(load(descs[i]),*(load(row['inputs'][k]) for k in ('z','a','target','mean_xy','std')))
    # Test-only metadata seam: the original full native metadata is already read
    # from its exact sealed file. Do not rebuild its original saved fullstrict.
    monkeypatch.setattr(m.native.allmode,'root_metadata',lambda _:deepcopy(original['source_roots']))
    p=m.assemble(a,manifest,report,receipt,original,data,arrays,descs)
    assert len(p['plans'])==6 and all(p['plans'][i]['status']=='sealed_raw15' for i in m.ACTIVE)
    assert [p['plans'][i]['raw_float32_sha256'] for i in m.ACTIVE]==[m.byte_sha(arrays[i][0]) for i in m.ACTIVE]


@pytest.mark.parametrize('failure',['write','fsync'])
def test_real_store_guard_persistence_failure_no_first_intent_or_callback(tmp_path,failure):
    _,s0,p,_,_=complete();plan=p['plans'][0]
    prefix=m.prefix_payload([e for e in s0.events if e.get('branch')==0])
    store=m.Store(tmp_path/(m.NAME+'_RAM_ONLY'));sink=m.RecordingSink(store)
    sink.prefix[0]=prefix
    meter=m.Meter(sink);row=m.rows_for(p)[0];meter.context=(row,plan,p)
    meter.where=dict(branch=0,root_slot=0,condition='off',arm=plan['arm'],selection_digest=plan['selection_digest'])
    called=[];write=store.write
    def broken(name,value,**kw):
        if name=='journal.jsonl' and json.loads(value)['kind']=='initial_history_guard':
            if failure=='write':raise OSError('RAM simulated durable write failure')
            with pytest.MonkeyPatch.context() as mp:
                def fail_fsync(_):raise OSError('RAM simulated fsync failure')
                mp.setattr(m.os,'fsync',fail_fsync)
                return write(name,value,**kw)
        return write(name,value,**kw)
    store.write=broken
    with pytest.raises(OSError):meter.call('public_step',lambda:called.append('BAD'),phase='plan',step=1)
    assert not called and meter.receipts==[] and meter.counts['public_step']==dict(attempted=0,completed=0)
    assert row['initial_history_guard'] is None and not meter.accepted
    journal=store.path/'journal.jsonl'
    if journal.exists():assert b'operation_dispatch_intent' not in journal.read_bytes()


def test_full_snapshot_not_just_state_and_goal_phase_semantics():
    _,s,p,_,_=complete();plan=p['plans'][0]
    prefix=m.prefix_payload([e for e in s.events if e.get('branch')==0])
    m.initial_proof(plan,p,prefix)
    obs=[e for e in prefix if e['kind']=='fresh_observation']
    assert obs[30]['pre']['goal_state']!=obs[33]['pre']['goal_state']
    assert obs[30]['receipt']['snapshot']==obs[33]['receipt']['snapshot']
    # Same physical state7 but wrong full physical snapshot after binding rejects.
    q=deepcopy(prefix);last=[e for e in q if e['kind']=='fresh_observation'][-1]
    for snap in (last['receipt']['snapshot'],last['pre']['snapshot'],last['post']['snapshot']):
        snap['state_float64_bits']='bad full snapshot, state7 unchanged'
    assert last['receipt']['state']==obs[-1]['receipt']['state']
    with pytest.raises(ValueError):m.initial_proof(plan,p,q)


@pytest.mark.parametrize('mutation',['missing3','duplicate1','wrong3branch'])
def test_noncontiguous_six_slot_chain_closure_mutations(mutation):
    r0,s,p,_,_=complete();r,ev=deepcopy(r0),deepcopy(s.events)
    closures=[e for e in ev if e['kind']=='branch_closed']
    assert [e['receipt']['index'] for e in closures]==[0,1,3,4]
    if mutation=='missing3':ev.remove(closures[2])
    elif mutation=='duplicate1':closures[2]['receipt']=deepcopy(closures[1]['receipt'])
    else:
        next(e for e in ev if e.get('branch')==3)['branch']=2
    reseal(ev,r)
    with pytest.raises(ValueError):m.validate_saved(r,ev,p)
