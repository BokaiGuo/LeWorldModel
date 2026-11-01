"""RAM-only array mocks and saved byte receipts; never neural/native physics."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import agent_allmode_physical_20261028_v1 as m
from test_p1_moving_overlay_recompile_physical_20261027_v1 import fixture as old_fixture, Env, REQUEST


def h(value):return hashlib.sha256(str(value).encode()).hexdigest()


class MemoryStore(m.Store):
    def __init__(self):
        self.path=Path('/RAM_ONLY');self.files={};self.events=[];self.previous=None;self.index=0
        self.free=m.FLOOR+m.CAP
    def check(self,size=4096,final=False):
        m.require(sum(map(len,self.files.values()))+size<=m.CAP-(0 if final else m.FINAL) and
                  self.free>=m.FLOOR+size,'RAM cap/floor')
    def write(self,name,value,append=False,final=False):
        self.check(len(value),final)
        assert append or name not in self.files
        self.files[name]=self.files.get(name,b'')+value if append else value
        return hashlib.sha256(self.files[name]).hexdigest()
    def emit(self,kind,**fields):
        before=len(self.files.get('journal.jsonl',b''));super().emit(kind,**fields)
        self.events.append(json.loads(self.files['journal.jsonl'][before:]))


@lru_cache(maxsize=1)
def _fixture():
    _,data,_=old_fixture();slots=[];plans=[];rows=[]
    for root in data['roots']:
        e=Env();e.reset(seed=root['slot']['reset_seed'])
        obs,_=e.reset(seed=0,options=dict(state=REQUEST.copy(),goal_state=e.goal_state.copy()))
        trace=[]
        for step in range(11):
            if step:obs,*_=e.step(m.moving.warmup_commands()[step-1].copy())
            for overlay,flag in enumerate((False,True,False)):
                e.with_target=flag;pre=m.invariant(e);a=e.render();post=m.invariant(e)
                buf=io.BytesIO();Image.fromarray(a).save(buf,format='PNG')
                item=dict(receipt=dict(phase='warmup',step=step,condition='on' if flag else 'off',
                    overlay_index=overlay,file_sha256=hashlib.sha256(buf.getvalue()).hexdigest(),
                    pixel_sha256=m.moving.image_sha(a),state=obs['state'].tolist(),snapshot=post['snapshot']),pre=pre,post=post)
                if step in (0,5,10) or overlay==0:trace.append(item)
            e.with_target=False
        root['warm_trace']=trace
    for index,(slot,condition) in enumerate(m.ORDER):
        export=data['exports'][slot,condition]
        s=dict(index=index,root_slot=slot,condition=condition,task_id=h(('task',slot,condition)),
               blind_sha256=export['sha256'],task_sha256=h(('taskfile',index)),task_file='/RAM/task'+str(index))
        slots.append(s)
        # Include identical/no-op plans: never select only changed actions.
        commands=np.full((10,2),np.float32(.25 if index%2 else 0),np.float32).tolist()
        p=dict(schema=m.PRODUCER+'.model_plan',root_slot=slot,source_slot=slot,condition=condition,arm=m.ARM,
               task_id=s['task_id'],blind_sha256=s['blind_sha256'],task_sha256=s['task_sha256'],source_report_sha256=m.SOURCE_SHA,
               commands=commands,prefix_commands=commands[:5],suffix_commands=commands[5:],selected_index=index,
               objective=.3,model_Q=.4,horizon=2,raw_steps=10,selection_uses_physics=False,
               P1_admission=False,P3_admission=False,equal_total_compute=False)
        p['selection_digest']=m.core.digest(p);rows.append(dict(index=index,root_slot=slot,condition=condition,status='plan_sealed',plan=deepcopy(p)))
        plans.append(dict(p,whole_run_report_sha256=h('actual compiler')))
    compiled=dict(schema=m.PRODUCER+'.prepare',source_report_sha256=m.SOURCE_SHA,slots=slots,bindings={})
    report=dict(schema=m.PRODUCER+'.report',status='all4_allmode_plans_sealed_not_admitted',errors=[],first_error=None,
        commands_released=True,model_unchanged=True,source_verified=True,P1_admission=False,P3_admission=False,rows=rows,
        resources={k:dict(attempted=1032,completed=1032) for k in ('predict','action_encoder','pred_proj')})
    consumed=dict(schema=m.PRODUCER+'.consume',status='all4_allmode_commands_verified_not_admitted',errors=[],plans=plans,
        P1_admission=False,P3_admission=False,equal_total_compute=False,new_neural_calls=0,new_physics_calls=0)
    p=dict(schema=m.NAME+'.prepare',arguments=dict(producer_output='/RAM/compiler',producer_report_sha256=h('actual compiler'),
        producer_source_sha256=h('RAM v2 source'),producer_prepared='/RAM/v2_prepare.json',producer_prepare_sha256=h('RAM v2 prepare')),
        contract=m.contract(),plans=plans,source_slots=[deepcopy(r['slot']) for r in data['roots']],
        warm_traces=[deepcopy(r['warm_trace']) for r in data['roots']],source_report_sha256=m.SOURCE_SHA,
        source_roots=m.root_metadata(data),
        producer_report_sha256=h('actual compiler'),producer_prepare_sha256=h('RAM v2 prepare'),
        producer_consume=consumed,bindings={})
    return p,data,report,compiled,consumed


def fixture():return deepcopy(_fixture())


def fake_place(env,plan,slot,meter,store):
    _,info=meter.call('reset',lambda:env.reset(seed=slot['reset_seed']),role='seed',seed=slot['reset_seed'])
    goal=info['goal_state'].copy()
    store.emit('placement_recipe',**meter.where,slot=slot,native_goal=goal.tolist(),requested_state=REQUEST.tolist())
    obs,_=meter.call('reset',lambda:env.reset(seed=0,options=dict(state=REQUEST.copy(),goal_state=goal.copy())),role='placement',seed=0)
    return obs,goal


def execute(monkeypatch,*,p=None,data=None,factory=Env,store=None,started=None):
    fp,fd,*_=fixture();p=fp if p is None else p;data=fd if data is None else data
    store=MemoryStore() if store is None else store;envs=[]
    def make():e=factory();envs.append(e);return e
    monkeypatch.setattr(m,'place',fake_place);monkeypatch.setattr(m,'validate_env',lambda _:None)
    monkeypatch.setattr(m,'sha',lambda path:hashlib.sha256(store.files[Path(path).name]).hexdigest())
    r=m.execute(p,data,store,make,authorization=m.AUTH,prepared='/RAM/physical_prepare.json',
                prepare_sha256=h('physical prepare'),started=started)
    return r,store,envs,p


@lru_cache(maxsize=1)
def completed():
    with pytest.MonkeyPatch.context() as mp:return execute(mp)


def test_full4_exact_raw10_history_native_counts_and51_bridge():
    r,s,envs,p=completed()
    assert r['status']==m.RECORDED,r['errors']
    assert len(r['rows'])==len(envs)==r['fixed_denominator']==4
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()}
    assert r['new_neural_calls']==r['fresh_roots']==r['new_independent_samples']==0
    assert len(m.validate_saved(r,s.events,p))==4
    assert len([f for f in s.files if f.endswith('.png')])==264
    for row,env,plan in zip(r['rows'],envs,p['plans'],strict=True):
        assert env.closed and env.dispatch_flags==[False]*20 and env.with_target is False
        assert np.asarray(env.commands[:10],np.float32).tobytes()==m.moving.warmup_commands().tobytes()
        assert np.asarray(env.commands[10:],np.float32).tobytes()==m.raw10(plan['commands']).tobytes()
        assert row['own_attempted']==row['own_returned']==row['warmup_returned']==10
        assert row['prefix51']['complete'] is True and len(row['prefix51']['frames'])==51
        assert [cp['step'] for cp in row['checkpoints']]==[0,5,10] and row['missing_checkpoints']==[]
        assert '_set_state' not in env.__dict__ and 'reset' not in env.__dict__


@pytest.mark.parametrize('mutation',['partial','status','denominator','order','bool','condition','arm','task','blind','tasksha','source',
    'seal','rawULP','prefix','whole_report','resources','flags'])
def test_real_producer_ABI_mutations_failclosed(mutation):
    _,data,r,c,v=fixture();p=v['plans'][0]
    if mutation=='partial':r['commands_released']=False
    elif mutation=='status':v['status']='all4_allmode_plans_sealed_not_admitted'
    elif mutation=='denominator':v['plans'].pop()
    elif mutation=='order':v['plans'].reverse()
    elif mutation=='bool':p['root_slot']=False
    elif mutation=='condition':p['condition']='on'
    elif mutation=='arm':p['arm']='full10_conditional_D2P'
    elif mutation=='task':p['task_id']=h('bad')
    elif mutation=='blind':p['blind_sha256']=h('bad')
    elif mutation=='tasksha':p['task_sha256']=h('bad')
    elif mutation=='source':p['source_report_sha256']=h('bad')
    elif mutation=='seal':p['selection_digest']=h('bad')
    elif mutation=='rawULP':p['commands']=deepcopy(p['commands']);p['commands'][0][0]=float(np.nextafter(np.float32(0),np.float32(1)))
    elif mutation=='prefix':p['prefix_commands']=deepcopy(p['prefix_commands']);p['prefix_commands'][0][0]=.125
    elif mutation=='whole_report':p['whole_run_report_sha256']=h('bad')
    elif mutation=='resources':r['resources']['predict']['completed']=1031
    else:p['P1_admission']=True
    with pytest.raises((ValueError,KeyError)):m.validate_verified(r,c,v,data,h('actual compiler'))


@pytest.mark.parametrize('fault',['terminal','truncate'])
@pytest.mark.parametrize('at',[5,11,15,18])
def test_native_earlystop_no_padding_or51_fabrication(monkeypatch,fault,at):
    r,s,envs,p=execute(monkeypatch,factory=lambda:Env(fault=fault,at=at))
    # A terminal that changes a saved warmup state is a source-identity failure,
    # not a license to replace the fixed root. Truncation leaves it unchanged.
    mismatch=(at==5 and fault=='terminal')
    assert r['status']==('partial_or_failed' if mismatch else m.RECORDED),r['errors']
    assert all(e.closed and len(e.commands)==at for e in envs)
    if not mismatch:assert len(m.validate_saved(r,s.events,p))==4
    for row in r['rows']:
        assert row['own_returned']==max(at-10,0)
        assert row['prefix51']['complete'] is (at>=15)
        assert 10 in row['missing_checkpoints']


@pytest.mark.parametrize('field',['file_sha256','pixel_sha256','state','snapshot','pre'])
def test_source_warmup_off_and_on_mismatch_blocks_after_png_retention(monkeypatch,field):
    p,data,*_=fixture();ref=next(x for x in p['warm_traces'][0] if x['receipt']['step']==5 and x['receipt']['overlay_index']==1)
    if field=='pre':ref['pre']['goal_pose'][0]+=1
    elif field=='state':ref['receipt']['state'][0]+=1e-8
    elif field=='snapshot':ref['receipt']['snapshot']['state_float64_bits']='wrong'
    else:ref['receipt'][field]=h('wrong')
    data['roots'][0]['warm_trace']=deepcopy(p['warm_traces'][0])
    r,s,envs,_=execute(monkeypatch,p=p,data=data)
    assert all(row['status']=='engineering_failed' and row['own_returned']==0 for row in r['rows'][:2])
    assert all(row['status']=='horizon_completed' for row in r['rows'][2:])
    assert r['rows'][0]['last_observation']['png'] in s.files


@pytest.mark.parametrize('fault',['throw','substitute','guard','render'])
def test_failed_return_incident_primary_secondary_and_four_denominator(monkeypatch,fault):
    r,s,envs,_=execute(monkeypatch,factory=lambda:Env(fault=fault,at=11,close_error=True))
    assert r['status']=='partial_or_failed' and len(r['rows'])==4
    assert all(e.closed for e in envs)
    if fault=='substitute':
        # A zero requested action returned as zero is NOT a substitution.
        assert [len(e.commands) for e in envs]==[20,11,20,11]
    else:assert all(len(e.commands)==11 for e in envs)
    assert all(row['first_error'] for row in r['rows'])
    affected=r['rows'][1::2] if fault=='substitute' else r['rows']
    assert all(row['secondary_errors'] for row in affected)
    if fault=='throw':assert r['counts']['public_step']==dict(attempted=44,completed=40)


def reseal(events,report):
    tail=None
    for i,e in enumerate(events):
        e.update(index=i,previous_sha256=tail);e['event_sha256']=m.source.digest({k:v for k,v in e.items() if k!='event_sha256'});tail=e['event_sha256']
    report['journal_chain_tail']=tail


@pytest.mark.parametrize('field',['denominator','rowkey','raw','sourceON','goal','bridge','metric','delta','reward','native_stop','checkpoint','count','chain','unknown','operation_scope','renderer','bound_goalpose'])
def test_strict_saved_consumer_mutations_even_resealed(field):
    r0,s,_,p0=completed();r,p,events=deepcopy(r0),deepcopy(p0),deepcopy(s.events)
    if field=='denominator':r['fixed_denominator']=3
    elif field=='rowkey':r['rows'][0]['condition']='on'
    elif field=='raw':next(e for e in events if e['kind']=='operation_dispatch_intent' and e['receipt']['resource']=='public_step' and e['receipt'].get('phase')=='plan')['receipt']['raw_bits']='wrong'
    elif field=='sourceON':next(e for e in events if e['kind']=='fresh_observation' and e['receipt']['overlay_index']==1)['receipt']['pixel_sha256']=h('bad')
    elif field=='goal':r['rows'][0]['goal_pose'][0]+=1
    elif field=='bridge':r['rows'][0]['prefix51']['frames'][0]['receipt']['pixel_sha256']=h('bad')
    elif field=='metric':r['rows'][0]['responses'][0]['native_metrics']['native_distance']=0.
    elif field=='delta':r['rows'][0]['responses'][0]['delta_native7'][0]+=1
    elif field=='reward':r['rows'][0]['responses'][0]['reward']=0.
    elif field=='native_stop':r['rows'][0]['responses'][0]['terminated']=True
    elif field=='checkpoint':r['rows'][0]['checkpoints'][-1]['state'][0]+=1
    elif field=='count':r['counts']['physics_step']['completed']-=1
    elif field=='chain':events[2]['event_sha256']=h('bad')
    elif field=='operation_scope':
        for e in events:
            if e['kind'].startswith('operation_') and e['receipt']['resource']=='constructor' and e['receipt']['branch']==0:e['receipt']['condition']='on'
        r['actual_operation_receipts'][0]['condition']='on'
    elif field=='renderer':
        e=next(e for e in events if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan')
        e['pre']['renderer']['action_scale']=1;e['post']['renderer']['action_scale']=1
    elif field=='bound_goalpose':
        r['rows'][0]['goal_pose'][0]+=1
        next(e for e in events if e['kind']=='goal_attached' and e['branch']==0)['goal_pose']=deepcopy(r['rows'][0]['goal_pose'])
    else:r['new_independent_samples']=1
    if field!='chain':reseal(events,r)
    with pytest.raises((ValueError,KeyError)):m.validate_saved(r,events,p)


def test_authorization_before_verify_or_factory(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('unauthorized read'))
    with pytest.raises(ValueError):m.run('/RAM/p',h('p'),'/RAM/o',lambda:pytest.fail('factory'),authorization='wrong')


@pytest.mark.parametrize('value',['0'*64,'x'*64,None])
def test_no_future_or_missing_hash(value):
    with pytest.raises(ValueError):m.actual_hash(value)


@pytest.mark.parametrize('field',['report','consume','prepare'])
def test_v1_schema_never_passes_v2_boundary(field):
    _,data,r,p,v=fixture();target={'report':r,'consume':v,'prepare':p}[field]
    target['schema']=target['schema'].replace('20261028_v2','20261028_v1')
    with pytest.raises(ValueError):m.validate_verified(r,p,v,data,h('actual compiler'))


def material_boundary(monkeypatch,tmp_path):
    _,data,r,p,v=fixture();calls=[];reads=[];directory=tmp_path/'v2_cuda';directory.mkdir()
    pp=str(tmp_path/'v2_prepare.json');rp=str(directory/'report.json')
    code=str(m.ROOT/'experiments/spectral_reachability'/(m.PRODUCER+'.py'))
    args=dict(producer_output=str(directory),producer_report_sha256=h(rp),producer_source_sha256=h(code),
              producer_prepared=pp,producer_prepare_sha256=h(pp))
    parent=str(m.ROOT/'results/p1_moving_overlay_recompile_cuda_20261027_v1_prepare_MAIN01.json')
    p.update(contract={'RAM':'contract'},bindings={parent:h(parent)})
    r.update(prepared=pp,prepare_sha256=h(pp),contract=p['contract'])
    for plan in v['plans']:plan['whole_run_report_sha256']=h(rp)
    hashes={str(m.base.__file__):m.BASE_SHA,**{s['task_file']:s['task_sha256'] for s in p['slots']}}
    docs={rp:r,pp:p,parent:dict(bindings={'/RAM/weight':h('/RAM/weight')})}
    def read(path,cap=m.PREPARE_CAP):reads.append((str(path),cap));return deepcopy(docs[str(path)])
    def consume(output,expected):calls.append((str(output),expected));return deepcopy(v)
    module=SimpleNamespace(__file__=code,consume=consume,contract=lambda:p['contract'])
    monkeypatch.setattr(m,'sha',lambda f:hashes.get(str(f),h(str(f))))
    monkeypatch.setattr(m,'read',read);monkeypatch.setattr(m,'source_material',lambda:deepcopy(data))
    monkeypatch.setattr(m,'own_files',lambda:[Path('/RAM/adapter.py')])
    monkeypatch.setattr(m,'constructor_contract',lambda:('/RAM/native.py',h('/RAM/native.py')))
    monkeypatch.setattr(m,'PRODUCER_SHA',h(code));monkeypatch.setattr(m,'PRODUCER_PREP',Path(pp));monkeypatch.setattr(m,'PRODUCER_PREP_SHA',h(pp))
    monkeypatch.setattr(m.importlib,'import_module',lambda _:module)
    return args,calls,reads,module,hashes,docs


def test_v2_complete_consume_and_external_prepare_report_source_seals(monkeypatch,tmp_path):
    args,calls,reads,module,hashes,docs=material_boundary(monkeypatch,tmp_path)
    p,_=m.material(args)
    assert calls==[(args['producer_output'],args['producer_report_sha256'])]
    assert p['producer_prepare_sha256']==args['producer_prepare_sha256']
    assert len(p['plans'])==4 and p['producer_consume']['plans']==p['plans']
    assert '/RAM/weight' in p['bindings'] and '/RAM/native.py' in p['bindings']
    assert (str(Path(args['producer_output'])/'report.json'),512*1024**2) in reads


@pytest.mark.parametrize('fault',['source','helper','report','prepare','partial','consume','drift'])
def test_external_material_tamper_never_bypasses_consume(monkeypatch,tmp_path,fault):
    args,calls,reads,module,hashes,docs=material_boundary(monkeypatch,tmp_path)
    if fault=='source':args['producer_source_sha256']=h('wrong')
    elif fault=='helper':hashes[str(m.base.__file__)]=h('wrong')
    elif fault=='report':args['producer_report_sha256']=h('wrong')
    elif fault=='prepare':args['producer_prepare_sha256']=h('wrong')
    elif fault=='partial':docs[str(Path(args['producer_output'])/'report.json')]['status']='partial_or_failed'
    elif fault=='consume':
        def fail(*a):raise ValueError('real consume failed')
        module.consume=fail
    else:
        original=module.consume
        def drift(*a):
            result=original(*a);hashes[str(Path(args['producer_output'])/'report.json')]=h('drift');return result
        module.consume=drift
    with pytest.raises(ValueError):m.material(args)
    if fault in ('source','helper','report','prepare','partial'):assert calls==[]


def test_actual_saved_consume_checks_all_png_pixels_closure_and_retains_four(monkeypatch):
    r,s,_,p=completed();files=deepcopy(s.files)
    monkeypatch.setattr(m,'verify',lambda *a:(deepcopy(p),{}))
    monkeypatch.setattr(m,'saved_artifacts',lambda _:deepcopy(files))
    out=m.consume('/RAM_ONLY',hashlib.sha256(files['report.json']).hexdigest())
    assert out['fixed_denominator']==len(out['rows'])==4 and out['rows']==r['rows']
    assert out['errors']==[] and out['new_neural_calls']==out['new_physics_calls']==0
    assert len(out['bindings'])==len(files)+1 and not out['P1_admission']


@pytest.mark.parametrize('fault',['pngbytes','extra','binding','journal','external','finaldrift'])
def test_saved_consume_byte_tamper_failclosed(monkeypatch,fault):
    _,s,_,p=completed();files=deepcopy(s.files);expected=hashlib.sha256(files['report.json']).hexdigest();calls=[]
    monkeypatch.setattr(m,'verify',lambda *a:(deepcopy(p),{}))
    if fault=='pngbytes':files[next(f for f in files if f.endswith('.png'))]=b'tampered'
    elif fault=='extra':files['unaccounted.bin']=b'extra'
    elif fault=='binding':files['execution_binding.json']=b'{}'
    elif fault=='journal':files['journal.jsonl']=files['journal.jsonl'][:-1]
    elif fault=='external':expected=h('wrong')
    def saved(_):
        calls.append(1);result=deepcopy(files)
        if fault=='finaldrift' and len(calls)>1:result['late.bin']=b'drift'
        return result
    monkeypatch.setattr(m,'saved_artifacts',saved)
    with pytest.raises(ValueError):m.consume('/RAM_ONLY',expected)


def test_meter_entry_return_failure_no_refund_and_cleanup_after_wall(monkeypatch):
    store=MemoryStore();meter=m.Meter(store);called=[]
    def fail():called.append('entered');raise RuntimeError('callback failure')
    with pytest.raises(RuntimeError):meter.call('public_step',fail)
    assert meter.counts['public_step']==dict(attempted=1,completed=0)
    emit=store.emit
    def fail_return(kind,**fields):
        if kind=='operation_returned':raise OSError('return journal')
        return emit(kind,**fields)
    store.emit=fail_return
    with pytest.raises(OSError):meter.call('public_step',lambda:called.append('returned'))
    assert meter.counts['public_step']==dict(attempted=2,completed=1) and meter.fault
    with pytest.raises(ValueError):meter.call('public_step',lambda:pytest.fail('must not enter'))
    meter.call('close',lambda:called.append('close'),cleanup=True)
    assert called==['entered','returned','close'] and meter.secondary


def test_intent_failure_and_caps_floor_refuse_before_callback():
    store=MemoryStore();meter=m.Meter(store)
    store.free=m.FLOOR-1
    with pytest.raises(ValueError):meter.call('constructor',lambda:pytest.fail('floor'))
    assert meter.counts['constructor']==dict(attempted=0,completed=0)
    store.free=m.FLOOR+m.CAP
    def fail(*a,**kw):raise OSError('intent journal')
    store.emit=fail
    with pytest.raises(OSError):meter.call('constructor',lambda:pytest.fail('intent'))
    assert meter.counts['constructor']==dict(attempted=0,completed=0)
    meter=m.Meter(MemoryStore());meter.counts['constructor']['attempted']=4
    with pytest.raises(ValueError):meter.call('constructor',lambda:pytest.fail('cap'))


def test_expired_and_partial_constructor_preserve_four_unknown(monkeypatch):
    r,s,envs,p=execute(monkeypatch,started=m.time.monotonic()-m.WALL-1)
    assert not envs and r['status']=='partial_or_failed' and len(r['rows'])==4
    assert all(row['native_success'] is row['final_state'] is None for row in r['rows'])
    assert r['counts']['constructor']==dict(attempted=0,completed=0)
    assert len(m.validate_saved(r,s.events,p))==4
    def fail():raise RuntimeError('constructor failure')
    r,_,_,_=execute(monkeypatch,factory=fail)
    assert r['counts']['constructor']==dict(attempted=4,completed=0)
    assert r['total_internal_physics_scope']=='unknown_constructor_work'


@pytest.mark.parametrize('fault',['zero','scale','order','dtype','reshape'])
def test_original_nonzero_warmup_not_arbitrary_raw_history(monkeypatch,fault):
    p,data,*_=fixture()
    for packet in data['roots'][0]['packets'].values():
        value=packet['raw_action_history']
        if fault=='zero':value=np.zeros_like(value)
        elif fault=='scale':value=value*np.float32(100)
        elif fault=='order':value=value[::-1].copy()
        elif fault=='dtype':value=value.astype(np.float64)
        else:value=value.reshape(5,4)
        packet['raw_action_history']=value
    r,_,envs,_=execute(monkeypatch,p=p,data=data)
    assert all(row['status']=='engineering_failed' for row in r['rows'][:2])
    assert all(not env.resets for env in envs[:2])
    assert all(row['status']=='horizon_completed' for row in r['rows'][2:])


def test_no_old_module_globals_changed_by_new_executor(monkeypatch):
    names=('NAME','ARM','ARMS','contract','execute','episode','capture','place','old_image','native_hooks','Meter')
    before={name:getattr(m.base,name,None) for name in names}
    r,_,_,_=execute(monkeypatch)
    assert r['status']==m.RECORDED
    assert before=={name:getattr(m.base,name,None) for name in names}


def test_cli_never_constructs_without_authorization(monkeypatch):
    monkeypatch.setattr(m,'run',lambda *a,**kw:pytest.fail('no run without MAIN authorization'))
    with pytest.raises(ValueError):m.main(['run','--prepared','/RAM/p','--prepare-sha',h('p'),'--output','/RAM/o'])


def test_store_new_schema_exclusive_cap_floor_and_symlink(tmp_path,monkeypatch):
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.FLOOR+m.CAP+1024**2))
    directory=tmp_path/(m.NAME+'_RAM_only');store=m.Store(str(directory))
    store.put('metadata.json',{'RAM':True})
    with pytest.raises(FileExistsError):store.put('metadata.json',{'RAM':False})
    with pytest.raises(ValueError):m.Store(str(directory))
    with pytest.raises(ValueError):m.Store(str(tmp_path/'not_owned'))
    with pytest.raises(ValueError):store.check(m.CAP)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.FLOOR-1))
    with pytest.raises(ValueError):store.check()
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.FLOOR+m.CAP+1024**2))
    (directory/'symlink').symlink_to(directory/'metadata.json')
    with pytest.raises(ValueError):store.check()


def test_prepare_verify_exclusive_metadata_reconstruction_only(monkeypatch,tmp_path):
    p,data,*_=fixture();path=tmp_path/(m.NAME+'_prepare_RAM.json')
    monkeypatch.setattr(m,'material',lambda args:(deepcopy(p),deepcopy(data)))
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.FLOOR+m.CAP))
    result=m.prepare(str(path),p['arguments'])
    assert m.verify(str(path),result['prepare_sha256'])[0]==p
    with pytest.raises(ValueError):m.prepare(str(path),p['arguments'])
    value=json.loads(path.read_bytes());value['extra_field']='tampered'
    path.write_bytes(m.raw(value))
    with pytest.raises(ValueError):m.verify(str(path),m.sha(path))


def reuse_fixture():
    p,data,*_=fixture();oldplans=[];oldrows=[]
    for slot in (0,1):
        for condition in ('off','on'):
            for arm in ('line_search_GN','full10_conditional_D2P'):
                new=p['plans'][2*slot+(condition=='on')]
                original=dict(root_slot=slot,condition=condition,arm=arm,task_id=new['task_id'],
                    blind_sha256=new['blind_sha256'],source_report_sha256=m.SOURCE_SHA,
                    commands=deepcopy(new['commands']),selection_digest=h((slot,condition,arm)))
                oldplans.append(original)
                for variant in ('open_full','teacher_full','teacher_small'):
                    task=p['source_roots'][slot]
                    plan=dict(original_plan=deepcopy(original),commands=deepcopy(original['commands']))
                    oldrows.append(dict(index=len(oldrows),root_slot=slot,condition=condition,arm=arm,variant=variant,
                        task_id=original['task_id'],selection_digest=original['selection_digest'],
                        selected_suffix=dict(plan=plan),status='suffix_completed',first_error=None,secondary_errors=[],
                        start_verified=True,**{k:deepcopy(task[k]) for k in ('root_state','root_sha256','goal','goal_pose')},
                        goal_pixel_sha256=task['goal_pixels'][condition],native_success=False))
    p['bindings']={str(m.base.SOURCE_REPORT):m.SOURCE_SHA}
    for i,plan in enumerate(p['plans']):p['bindings']['/RAM/task'+str(i)]=plan['task_sha256']
    old=dict(schema=m.OLD_PHYSICAL+'.consume',status='factorial24_suffix_physical_recorded_not_admitted',
        rows=oldrows,fixed_denominator=24,bindings=deepcopy(p['bindings']),physical_report_sha256=h('old physical'),
        P1_admission=False,scientific_admission=False,independent_confirmation=False,new_neural_calls=0)
    return p,old,dict(plans=oldplans)


def test_reuse_full_consumed24_retains_four_and_false_freshness():
    p,old,prepared=reuse_fixture();rows=m.reuse_rows(p,old,prepared)
    assert len(rows)==4 and [r['old_index'] for r in rows]==[4,10,16,22]
    assert all(r['status']=='inherited_saved_physical_not_fresh' for r in rows)
    assert all(r['freshness'] is r['P1_admission'] is False for r in rows)


@pytest.mark.parametrize('fault',['condition','task','source','blind','root','goal','pose','pixels','selection','duplicate','taskfile'])
def test_reuse_ancestry_conflicts_failclosed(fault):
    p,old,prepared=reuse_fixture();r=old['rows'][4]
    if fault=='condition':r['condition']='on'
    elif fault=='task':r['task_id']=h('bad')
    elif fault=='source':old['bindings'][str(m.base.SOURCE_REPORT)]=h('bad')
    elif fault=='blind':prepared['plans'][1]['blind_sha256']=h('bad')
    elif fault=='root':r['root_state'][0]+=1
    elif fault=='goal':r['goal'][0]+=1
    elif fault=='pose':r['goal_pose'][0]+=1
    elif fault=='pixels':r['goal_pixel_sha256']=h('bad')
    elif fault=='selection':r['selection_digest']=h('bad')
    elif fault=='duplicate':old['rows'][10]=deepcopy(r)
    else:old['bindings']['/RAM/task0']=h('bad')
    with pytest.raises((ValueError,KeyError)):m.reuse_rows(p,old,prepared)


@pytest.mark.parametrize('fault',['different_new_raw','different_old_raw','failed','not_started'])
def test_reuse_unknown_does_not_filter_or_zero(fault):
    p,old,prepared=reuse_fixture();r=old['rows'][4]
    if fault=='different_new_raw':p['plans'][0]['commands'][0][0]=.5
    elif fault=='different_old_raw':r['selected_suffix']['plan']['commands'][0][0]=.5
    else:r['status']='engineering_failed' if fault=='failed' else 'not_started'
    rows=m.reuse_rows(p,old,prepared)
    assert len(rows)==4 and rows[0]['status']=='unknown' and rows[0]['evidence'] is None
    assert rows[0]['native_success'] is None and rows[1]['status']=='inherited_saved_physical_not_fresh'


def reuse_boundary(monkeypatch):
    p,old,prepared=reuse_fixture();calls=[]
    oldpath='/RAM/old24';rp=oldpath+'/report.json';pp='/RAM/old24_prepare.json'
    code=str(m.ROOT/'experiments/spectral_reachability'/(m.OLD_PHYSICAL+'.py'))
    old['bindings'][pp]=h(pp);old['physical_report_sha256']=h(rp)
    report=dict(schema=m.OLD_PHYSICAL+'.report',rows=deepcopy(old['rows']),errors=[],
        execution_binding=dict(prepared=pp,prepare_sha256=h(pp)))
    docs={rp:report,pp:prepared};hashes={code:m.OLD_PHYSICAL_SHA,rp:h(rp),**p['bindings'],pp:h(pp)}
    def material(args):calls.append('new');assert args==p['arguments'];return deepcopy(p),None
    def consume(output,expected):calls.append('old');assert str(output)==oldpath and expected==h(rp);return deepcopy(old)
    module=SimpleNamespace(__file__=code,consume=consume)
    monkeypatch.setattr(m,'material',material);monkeypatch.setattr(m,'read',lambda path,cap=m.PREPARE_CAP:deepcopy(docs[str(path)]))
    monkeypatch.setattr(m,'sha',lambda path:hashes[str(path)])
    monkeypatch.setattr(m.importlib,'import_module',lambda name:module)
    return p,old,report,hashes,module,calls,(p['arguments'],oldpath,h(rp),m.OLD_PHYSICAL_SHA)


def test_reuse_real_consume_ABI_both_full_results_retained(monkeypatch):
    p,old,report,hashes,module,calls,args=reuse_boundary(monkeypatch)
    r=m.reuse_material(*args)
    assert calls==['new','old'] and r['old_physical_consume']==old and r['producer_consume']==p['producer_consume']
    assert r['reused']==4 and r['unknown']==0 and r['old_report_metadata']['errors']==[]
    assert r['new_neural_calls']==r['new_physics_calls']==0 and r['freshness'] is r['P1_admission'] is False


@pytest.mark.parametrize('fault',['source','report','prepare','schema','old_consume','new_consume','drift'])
def test_reuse_boundaries_no_bypass_and_errors_propagate(monkeypatch,fault):
    p,old,report,hashes,module,calls,args=reuse_boundary(monkeypatch)
    if fault=='source':hashes[str(module.__file__)]=h('bad')
    elif fault=='report':hashes['/RAM/old24/report.json']=h('bad')
    elif fault=='prepare':hashes['/RAM/old24_prepare.json']=h('bad')
    elif fault=='schema':report['schema']='agent_observed_CEM_physical_20261028_v2.report'
    elif fault=='old_consume':module.consume=lambda *a:(_ for _ in ()).throw(ValueError('old real failure'))
    elif fault=='new_consume':monkeypatch.setattr(m,'material',lambda a:(_ for _ in ()).throw(ValueError('new real failure')))
    else:hashes['/RAM/task0']=h('bad')
    with pytest.raises(ValueError):m.reuse_material(*args)
    if fault in ('source','report','prepare','schema','new_consume'):assert calls==[]


def test_reuse_exclusive_bounded_persistent_receipt_and_reverification(monkeypatch,tmp_path):
    p,old,_=reuse_fixture()
    result=dict(schema=m.NAME+'.reuse',arguments=p['arguments'],old_output='/RAM/old',old_report_sha256=h('report'),
        old_source_sha256=m.OLD_PHYSICAL_SHA,reused=0,unknown=4,rows=[dict(status='unknown') for _ in range(4)])
    calls=[]
    def material(*a):calls.append(a);return deepcopy(result)
    monkeypatch.setattr(m,'reuse_material',material)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.FLOOR+m.CAP))
    path=tmp_path/(m.NAME+'_reuse_RAM.json');args=(p['arguments'],'/RAM/old',h('report'),m.OLD_PHYSICAL_SHA)
    receipt=m.reuse(str(path),*args)
    assert json.loads(path.read_bytes())==result and m.verify_reuse(str(path),receipt['sha256'])['unknown']==4
    before=len(calls)
    with pytest.raises(ValueError):m.reuse(str(path),*args)
    assert len(calls)==before
    tampered=deepcopy(result);tampered['rows']=[];path.write_bytes(m.raw(tampered))
    with pytest.raises(ValueError):m.verify_reuse(str(path),m.sha(path))
    monkeypatch.setattr(m,'CAP',1)
    with pytest.raises(ValueError):m.reuse(str(tmp_path/(m.NAME+'_cap.json')),*args)


def test_reuse_native_earlystop_receipt_is_preserved_not_padded():
    p,old,prepared=reuse_fixture();r=old['rows'][4]
    r.update(status='native_stopped',native_success=True,suffix_returned=2,
        checkpoints=[dict(step=0),dict(step=5)],missing_checkpoints=[10],responses=[dict(step=1),dict(step=2)])
    rows=m.reuse_rows(p,old,prepared)
    assert rows[0]['status']=='inherited_saved_physical_not_fresh' and rows[0]['evidence']==r
    assert rows[0]['evidence']['missing_checkpoints']==[10] and rows[0]['evidence']['suffix_returned']==2


def test_cli_saved_reuse_and_verify_do_not_dispatch_run(monkeypatch,capsys):
    calls=[]
    monkeypatch.setattr(m,'reuse',lambda *a:(calls.append(('reuse',a)) or dict(status='saved_only')))
    monkeypatch.setattr(m,'verify_reuse',lambda *a:(calls.append(('verify',a)) or dict(status='saved_only')))
    monkeypatch.setattr(m,'run',lambda *a,**kw:pytest.fail('no physical dispatch'))
    assert m.main(['reuse','--output','/RAM/receipt','--producer-output','/RAM/new','--old-output','/RAM/old',
        '--old-source-sha',m.OLD_PHYSICAL_SHA,'--old-report-sha',h('old')])==0
    assert calls[0][0]=='reuse' and calls[0][1][2:]==('/RAM/old',h('old'),m.OLD_PHYSICAL_SHA)
    assert m.main(['verify-reuse','--output','/RAM/receipt','--report-sha',h('receipt')])==0
    assert calls[1]==('verify',('/RAM/receipt',h('receipt')))
