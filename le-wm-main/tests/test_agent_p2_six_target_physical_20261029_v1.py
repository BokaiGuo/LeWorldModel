"""RAM native-shaped driver and saved ABI fixtures. No actual environment/NN/fullstrict."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_six_target_physical_20261029_v1 as m
from test_agent_p2_three_program_physical_20261029_v1 import fixture as old_fixture
from test_agent_allmode_physical_20261028_v1 import MemoryStore as OldMemoryStore, Env, fake_place,h,reseal


def actual_receipt():
    p=m.ACTUAL_PAID['replay_receipt'];assert m.sha(p)==m.ACTUAL_PAID['replay_receipt_sha']
    return json.loads(Path(p).read_bytes())


def args():return {k:h(k) if k.endswith('_sha') else '/RAM/'+k+'.json' for k in m.ARGS}


def fixture():
    original,data,*_=old_fixture();a=args();c=actual_receipt();c.update(manifest_sha256=a['replay_manifest_sha'],report_sha256=a['replay_report_sha'])
    mu=np.asarray([-.0087,.0068],np.float32);std=np.asarray([.2019,.2002],np.float32)
    past=np.divide(np.subtract(m.moving.warmup_commands(),mu,dtype=np.float32),std,dtype=np.float32).reshape(1,2,10)
    domain=dict(schema=m.paid.TAG+'.domain',authorities=deepcopy(c['controller_authorities']),controller_prepare=dict(slots=[]),
        rows=[],normalizer=dict(mean_bits=mu.tobytes().hex(),std_bits=std.tobytes().hex()),blocked_slot_ids=[],denominator=dict(total=6,checked=6,missing=0))
    manifest=dict(schema=m.paid.TAG+'.prepare',domain=deepcopy(c['domain']))
    report=dict(schema=m.paid.TAG+'.report',completed=True,errors=[],manifest_sha256=a['replay_manifest_sha'],rows=[])
    arrays={};descs={}
    for i,(root,condition,arm) in enumerate(m.ORDER):
        s=original['plans'][2*root+1]['planner_plan'];s['root_context']['action_history']=past[0].tolist()
        z=np.asarray(s['root_context']['latent_history'],np.float32)[None];target=np.full((1,192),np.float32(.125+i),np.float32)
        rr=np.full((15,2),np.float32(.125*(i%3)),np.float32)
        identity={k:s[k] for k in ('task_id','task_sha256','blind_sha256')}
        vals=dict(z=z.copy(),a=past.copy(),target=target.copy())
        slot=dict(slot=i,root=root,condition=condition,arm=arm,**identity,controller_payload={k:v.tolist() for k,v in vals.items()},
            input_sha256={k:m.byte_sha(v) for k,v in vals.items()},source=dict(kind='RAM_only_no_science'))
        domain['controller_prepare']['slots'].append(slot)
        domain['rows'].append(dict(slot=i,root=root,condition=condition,arm=arm,task_identity=deepcopy(identity),checks=dict(array_bits=dict(raw=rr.tobytes().hex()))))
        report['rows'].append(dict(slot=i,root=root,condition=condition,arm=arm,status='returned'))
        arrays[i]=(rr.copy(),z.copy(),past.copy(),target.copy(),mu.copy(),std.copy())
        descs[i]=dict(file=f'slot{i}_array_99.npy',file_sha256=h(('raw',i)),root=root,shape=[15,2],dtype='<f4',data_sha256=m.byte_sha(rr),file_bytes=248)
    p=m.assemble(a,manifest,domain,report,c,original,data,arrays,descs)
    return p,data,(a,manifest,domain,report,c,original,arrays,descs)


def test_actual_paid_receipt_schema_and_protocol_source_pins():
    c=actual_receipt();m.paid_complete(c,m.ACTUAL_PAID)
    assert c['raw_deployment_authorized'] is False and c['physical_approval_required'] is True
    assert all(m.sha(p)==h for p,h in m.PINS.items())
    assert m.digest([1,2])!=m.paid.core.digest(json.dumps([1,2],separators=(',',':')).encode())


def test_fixed_six_new_plans_no_missing_and_native_budget():
    p,_,_=fixture();m.validate_execution(p)
    assert all(x['status']=='sealed_raw15' for x in p['plans']) and len(p['plans'])==6
    assert m.contract()['caps']==dict(constructor=6,close=6,reset=12,public_step=150,physics_step=1524,set_state=24,goal_setter=18,goal_render=12,goal_bind=6,explicit_render=486)
    assert (m.CAP,m.FINAL,m.FLOOR,m.META)==tuple(x*1024**2 for x in (128,16,200,128)) and m.WALL==900
    assert p['plans'][2]['arm']==p['plans'][5]['arm']=='arbitrary'


@pytest.mark.parametrize('fault',['schema','status','subset','blocked','API','tokens','old4','missingflag','flag','order','manifest','report','RAM','approval'])
def test_paid_receipt_mutations(fault):
    p,_,_=fixture();c=deepcopy(p['planner_authority']['receipt']);a=p['arguments']
    if fault=='schema':c['schema']=m.prior.PAID+'.consume'
    elif fault=='status':c['status']='ready'
    elif fault=='subset':c['denominator']['returned']=5
    elif fault=='blocked':c['local_checks'][2]['blocked']='blocked'
    elif fault=='API':c['predict_API']=24
    elif fault=='tokens':c['module_resources']['AE_tokens']=96
    elif fault=='old4':c['local_checks']=[c['local_checks'][i] for i in (0,1,3,4)]
    elif fault=='missingflag':del c['raw_deployment_authorized']
    elif fault=='flag':c['scientific_admission']=True
    elif fault=='order':c['local_checks'].reverse()
    elif fault=='manifest':c['manifest_sha256']='0'*64
    elif fault=='report':c['report_sha256']='0'*64
    elif fault=='RAM':c['evidence_backend']='RAM_fixture_not_evidence'
    else:c['physical_approval_required']=False
    with pytest.raises((ValueError,KeyError)):m.paid_complete(c,a)


@pytest.mark.parametrize('fault',['raw','signedzero','task','blind','z','a','target','normalizer','NPY','domain','missing','origin'])
def test_assemble_exact_source_bindings(fault):
    _,data,v=fixture();a,pm,d,r,c,o,ar,descs=deepcopy(v)
    if fault=='raw':ar[2][0][0,0]=.5
    elif fault=='signedzero':ar[0][0][0,0]=-0.
    elif fault=='task':d['controller_prepare']['slots'][2]['task_id']='wrong'
    elif fault=='blind':d['rows'][2]['task_identity']['blind_sha256']='wrong'
    elif fault=='z':ar[2][1][0,0,0]+=.1
    elif fault=='a':ar[2][2][0,0,0]+=.1
    elif fault=='target':ar[2][3][0,0]+=.1
    elif fault=='normalizer':ar[2][4][0]=.1
    elif fault=='NPY':descs[2]['file']='slot0_array_99.npy'
    elif fault=='domain':d['blocked_slot_ids']=[2]
    elif fault=='missing':del ar[2]
    else:del d['controller_prepare']['slots'][2]['source']
    with pytest.raises((ValueError,KeyError)):m.assemble(a,pm,d,r,c,o,data,ar,descs)


class MemoryStore(OldMemoryStore):
    def check(self,size=4096,final=False):
        m.require(sum(map(len,self.files.values()))+size<=m.CAP-(0 if final else m.FINAL),'RAM own128MiB cap')


def execute(*,fault=None,at=11,corrupt=None,write_failure=False,close_error=False):
    p,data,_=fixture();store=MemoryStore();envs=[]
    def factory():e=Env(fault=fault,at=at,close_error=close_error);envs.append(e);return e
    original_proof=m.initial_proof
    def proof(plan,prepared,events):
        if corrupt:
            events=deepcopy(events)
            if corrupt=='state':events[-1]['receipt']['state'][0]+=.01
            elif corrupt=='pixels':events[-1]['receipt']['pixel_sha256']=h('bad')
            elif corrupt=='raw':next(e for e in events if e['kind']=='returned_transition')['actual_bits']='00'
            elif corrupt=='goal':next(e for e in events if e['kind']=='goal_attached')['goal'][0]+=.01
            else:events.pop()
        return original_proof(plan,prepared,events)
    emit=store.emit
    def emit_guard(kind,**fields):
        if write_failure and kind=='initial_history_guard':raise OSError('RAM simulated guard fsync failure')
        return emit(kind,**fields)
    store.emit=emit_guard
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m,'place',fake_place);mp.setattr(m,'validate_env',lambda _:None);mp.setattr(m,'initial_proof',proof)
        mp.setattr(m,'sha',lambda path:hashlib.sha256(store.files[Path(path).name]).hexdigest())
        r=m.execute(p,data,store,factory,authorization=m.AUTH,prepared='/RAM/own.json',prepare_sha256=h('own'))
    return r,store,p,data,envs


@lru_cache(maxsize=1)
def complete():return execute()


def test_complete_new_six_RAM_lifecycle_and_strict486png():
    r,s,p,_,envs=complete();assert r['status']==m.RECORDED,r['errors']
    assert len(m.validate_saved(r,s.events,p))==6 and len(envs)==6
    assert all(e.closed and len(e.commands)==25 for e in envs)
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()}
    assert r['counts']['reset']['completed']==12
    assert len([n for n in s.files if n.endswith('.png')])==486
    assert [e['receipt']['index'] for e in s.events if e['kind']=='branch_closed']==list(range(6))
    for i,e in enumerate(envs):
        assert np.asarray(e.commands[10:],np.float32).tobytes()==m.raw15(p['plans'][i]['commands']).tobytes()
        guard=next(x for x in s.events if x['kind']=='initial_history_guard' and x['branch']==i)
        intent=next(x for x in s.events if x['kind']=='operation_dispatch_intent' and x['receipt']['branch']==i
            and x['receipt']['resource']=='public_step' and x['receipt'].get('phase')=='plan')
        assert guard['index']<intent['index'] and guard['proof']['schema']==m.NAME+'.initial_history_proof'


@pytest.mark.parametrize('corrupt',['state','pixels','raw','goal','missing'])
def test_guard_refusal_no_plan_intent_no_callback(corrupt):
    r,s,_,_,envs=execute(corrupt=corrupt)
    assert r['status']=='partial_or_failed' and all(len(e.commands)==10 and e.closed for e in envs)
    assert not any(x['kind']=='operation_dispatch_intent' and x['receipt'].get('phase')=='plan' for x in s.events)
    assert not any(x['kind']=='initial_history_guard' for x in s.events)


def test_guard_write_failure_absolutely_no_plan_callback():
    r,s,_,_,envs=execute(write_failure=True)
    assert r['status']=='partial_or_failed' and all(len(e.commands)==10 for e in envs)
    assert not any(x['kind']=='operation_dispatch_intent' and x['receipt'].get('phase')=='plan' for x in s.events)


@pytest.mark.parametrize('fault,at',[('truncate',5),('truncate',11),('terminal',25),('throw',11)])
def test_earlystop_failure_keeps_six_denominator_nullcp(fault,at):
    r,s,p,_,envs=execute(fault=fault,at=at)
    assert len(m.validate_saved(r,s.events,p))==6
    assert all(e.closed and len(e.commands)==at for e in envs)
    assert all((15 in x['missing_checkpoints'])==(at<25) for x in r['rows'])
    assert r['status']==('partial_or_failed' if fault=='throw' else m.RECORDED)
    panel=m.checkpoint_panel(r['rows']);assert len(panel['rows'])==len(panel['pairs'])==24
    if at<25:assert all(x['metrics'] is None for x in panel['rows'] if x['step']==15)


@pytest.mark.parametrize('fault',['unknown','late_guard','missing_guard','guard_payload','cp15','arbitrary','reward','raw','count','goal','prefix','close2'])
def test_resealed_journal_or_summary_tampering(fault):
    r0,s,p0,_,_=complete();r,p,ev=deepcopy(r0),deepcopy(p0),deepcopy(s.events)
    if fault=='unknown':ev.insert(-1,dict(kind='unknown'))
    elif fault=='late_guard':g=next(x for x in ev if x['kind']=='initial_history_guard');ev.remove(g);ev.insert(-2,g)
    elif fault=='missing_guard':ev.remove(next(x for x in ev if x['kind']=='initial_history_guard'))
    elif fault=='guard_payload':next(x for x in ev if x['kind']=='initial_history_guard')['proof']['proof']['raw_float32_sha256']=h('bad')
    elif fault=='cp15':r['rows'][0]['checkpoints'].pop()
    elif fault=='arbitrary':r['rows'][2]['status']='missing'
    elif fault=='reward':next(x for x in ev if x['kind']=='returned_transition')['reward']=0.
    elif fault=='raw':next(x for x in ev if x['kind']=='operation_dispatch_intent' and x['receipt']['resource']=='public_step')['receipt']['raw_bits']='00'
    elif fault=='count':r['counts']['reset']['completed']=8
    elif fault=='goal':next(x for x in ev if x['kind']=='goal_attached')['goal'][0]+=.01
    elif fault=='prefix':next(x for x in ev if x['kind']=='fresh_observation')['receipt']['pixel_sha256']=h('bad')
    else:ev.remove([e for e in ev if e['kind']=='branch_closed'][2])
    reseal(ev,r)
    with pytest.raises((ValueError,KeyError,StopIteration)):m.validate_saved(r,ev,p)


def test_full_six_png_consumer_from_RAM_saved_loop(monkeypatch):
    r,s,p,data,_=complete()
    monkeypatch.setattr(m,'saved_artifacts',lambda _:deepcopy(s.files));monkeypatch.setattr(m,'verify',lambda *a:(deepcopy(p),data))
    result=m.consume('/RAM/physical',hashlib.sha256(s.files['report.json']).hexdigest())
    assert result['saved_PNG_verified']==486 and result['fixed_denominator']==6 and len(result['checkpoint_panel']['pairs'])==24
    assert result['checkpoint_panel']['comparable_pairs']==24
    monkeypatch.setattr(m,'saved_artifacts',lambda _:dict(s.files,extra=b'bad'))
    with pytest.raises(ValueError):m.consume('/RAM/physical',hashlib.sha256(s.files['report.json']).hexdigest())


def test_full_panel_gains_axis_components_and_pair_directions():
    r,_,_,_,_=complete();p=m.checkpoint_panel(r['rows'])
    assert len(p['rows'])==len(p['pairs'])==24
    for item in p['rows']:
        assert item['metrics'] is not None
        s=item['metrics'];parts=item['native7_squared_components']
        assert parts=={k:s[k] for k in ('native7_squared','joint_xy_squared','raw_angle_squared','agent_velocity_squared')}
        if item['step']==0:
            assert all(x==0 for x in item['cp0_error_gain'].values()) and item['cp0_reward_gain']==0 and item['cp0_angle_change_wrapped']==0
        base=p['rows'][item['slot']*4]['metrics']
        assert item['cp0_reward_gain']==s['native7_reward']-base['native7_reward']
    for pair in p['pairs']:
        li=pair['left_slot']*4+m.CP.index(pair['step']);ri=pair['right_slot']*4+m.CP.index(pair['step'])
        assert set(pair['delta'])==set(m.independent.SCALARS)|{'native_success'}
        assert pair['delta']['joint_xy_error']==p['rows'][ri]['metrics']['joint_xy_error']-p['rows'][li]['metrics']['joint_xy_error']


@pytest.mark.parametrize('failure',['write','fsync'])
def test_real_store_guard_durable_failure_no_step1(tmp_path,failure):
    _,s,p,_,_=complete();plan=p['plans'][2];store=m.Store(tmp_path/(m.NAME+'_RAM'))
    sink=m.RecordingSink(store);sink.prefix[2]=m.prior.prefix_payload([e for e in s.events if e.get('branch')==2])
    meter=m.Meter(sink);row=m.rows_for(p)[2];meter.context=(row,plan,p)
    meter.where=dict(branch=2,root_slot=0,condition='off',arm=plan['arm'],selection_digest=plan['selection_digest'])
    called=[];write=store.write
    def broken(name,value,**kw):
        if name=='journal.jsonl' and json.loads(value)['kind']=='initial_history_guard':
            if failure=='write':raise OSError('RAM durable write failure')
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(m.os,'fsync',lambda _:(_ for _ in ()).throw(OSError('RAM fsync failure')))
                return write(name,value,**kw)
        return write(name,value,**kw)
    store.write=broken
    with pytest.raises(OSError):meter.call('public_step',lambda:called.append('BAD'),phase='plan',step=1)
    assert called==[] and meter.receipts==[] and not meter.accepted and row['initial_history_guard'] is None
    if (store.path/'journal.jsonl').exists():assert b'operation_dispatch_intent' not in (store.path/'journal.jsonl').read_bytes()


def test_snapshot_not_state_only_and_goal_binding_phase():
    _,s,p,_,_=complete();prefix=m.prior.prefix_payload([e for e in s.events if e.get('branch')==0]);m.initial_proof(p['plans'][0],p,prefix)
    obs=[e for e in prefix if e['kind']=='fresh_observation']
    assert obs[30]['pre']['goal_state']!=obs[33]['pre']['goal_state'] and obs[30]['receipt']['snapshot']==obs[33]['receipt']['snapshot']
    q=deepcopy(prefix);last=[e for e in q if e['kind']=='fresh_observation'][-1]
    for snap in (last['receipt']['snapshot'],last['pre']['snapshot'],last['post']['snapshot']):snap['state_float64_bits']='bad snapshot same state7'
    with pytest.raises(ValueError):m.initial_proof(p['plans'][0],p,q)


def test_actual_saved_six_ABI_assemble_only_no_fullstrict(monkeypatch):
    a=m.ACTUAL_PAID;report=m.paid.read(a['replay_report'],a['replay_report_sha']);c=actual_receipt()
    manifest=m.paid.read(a['replay_manifest'],a['replay_manifest_sha']);domain=m.paid.read(manifest['domain']['path'],manifest['domain']['sha256'])
    original=json.loads(m.prior.ORIGINAL.read_bytes());assert m.sha(m.prior.ORIGINAL)==m.prior.ORIGINAL_SHA
    roots=[]
    for root in (0,1):
        plan=original['plans'][2*root+1]['planner_plan'];packet=m.moving.load_blind(plan['blind_path'],plan['blind_sha256'])
        roots.append(dict(slot=deepcopy(original['source_slots'][root]),warm_trace=deepcopy(original['warm_traces'][root]),packets=dict(off=packet,on=packet)))
    data=dict(roots=roots);arrays={};descs={};folder=Path(a['replay_report']).parent
    for i,row in enumerate(report['rows']):
        tr=m.paid.read(folder/row['trace']['file'],row['trace']['file_sha256']);descs[i]=row['arrays'][tr['arithmetic'][1]['output']]
        load=lambda d:m.paid.six.load_array(folder,d)
        arrays[i]=(load(descs[i]),*(load(row['inputs'][k]) for k in ('z','a','target','mean_xy','std')))
    monkeypatch.setattr(m.native.allmode,'root_metadata',lambda _:deepcopy(original['source_roots']))
    p=m.assemble(a,manifest,domain,report,c,original,data,arrays,descs)
    assert len(p['plans'])==6 and all(x['status']=='sealed_raw15' for x in p['plans'])
    assert all(p['plans'][i]['raw_float32_sha256']==m.byte_sha(arrays[i][0]) for i in range(6))


def test_authority_before_all_io_and_factory(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('read'))
    with pytest.raises(ValueError):m.run('/RAM/p',h('p'),'/RAM/o',lambda:pytest.fail('env'),authorization='bad')


def test_import_no_torch_gym_physics():
    code=f'import sys; from experiments.spectral_reachability import {m.NAME}; assert not any(k in sys.modules for k in ("torch","gymnasium","pymunk","stable_worldmodel.envs"))'
    p=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,text=True,capture_output=True)
    assert p.returncode==0,p.stderr
