"""RAM-only GRASP physical ABI/lifecycle/bytes regressions; no native physics/NN."""
from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_history_grasp_physical_20261029_v1 as m
from experiments.spectral_reachability import agent_p2_grasp_raw15_prepare_20261029_v1 as inputs
from test_agent_allmode_physical_20261028_v1 import fixture as old_fixture, MemoryStore as OldMemoryStore, Env, fake_place, h, reseal
from test_agent_allmode_prefix_20261028_v1 import slot as model_slot


def fixture():
    _,data,*_=old_fixture()
    compiled=dict(schema=m.PRODUCER+'.prepare',expected_model_hash=h('model'),slots=[],bindings={})
    plans=[]
    for root in (0,1):
        old=model_slot(root*2);ex=data['exports'][root,'off'];packet=data['roots'][root]['packets']['off']
        slot=dict(root_slot=root,condition='off',task_id=h(('task',root)),
            task_file='/RAM/task'+str(root),task_sha256=h(('taskfile',root)),
            blind_pointer=dict(path=str(ex['path']),sha256=ex['sha256'],opened=False),
            warm_raw10=m.moving.warmup_commands().tolist(),
            pixel_sha256=[m.moving.image_sha(x) for x in packet['rgb_history']]+[m.moving.image_sha(packet['goal_rgb'])],
            model_input=dict(root_context=deepcopy(old['packet']['root_context']),goal=deepcopy(old['packet']['goal']),
                q=np.eye(192).tolist(),normalization=inputs.normalization()))
        compiled['slots'].append(slot)
        logits=np.full(30,.125*root,np.float64);commands=np.tanh(logits).astype(np.float32).reshape(15,2)
        plan=dict(schema=m.CONSUMER+'.model_plan',root_slot=root,condition='off',task_id=slot['task_id'],
            task_file=slot['task_file'],task_sha256=slot['task_sha256'],source_report_sha256=m.native.SOURCE_SHA,
            blind_pointer=deepcopy(slot['blind_pointer']),original_context=deepcopy(slot['model_input']['root_context']),
            goal=deepcopy(slot['model_input']['goal']),Q=np.eye(192).tolist(),normalization=inputs.normalization(),
            model_sha256=compiled['expected_model_hash'],raw_steps=15,commands=commands.tolist(),
            raw_float32_sha256=hashlib.sha256(commands.astype('<f4').tobytes()).hexdigest(),logits=logits.tolist(),
            terminal=deepcopy(slot['model_input']['root_context']),model_terminal_Q=.125,
            planner_report_sha256=h('planner report'),physical_release=False,official_reproduction=False,
            P1_admission=False,P2_admission=False)
        plan['selection_digest']=m.planner_digest(plan);plans.append(plan)
    consumed=dict(schema=m.CONSUMER+'.consume',status='history_joint30_variant_verified_not_admitted',
        plans=plans,planner_report_sha256=h('planner report'),prepare_sha256=h('planner prepare'),resources=dict(predict_API=5412),
        consumer_bindings={},fix_policy='exact parameter-times-zero at empty Adam state only',
        new_neural_calls=0,new_physics=0,physical_release=False,P1_admission=False,P2_admission=False,
        official_reproduction=False,equal_budget_to_1902=False,errors=[])
    args=dict(planner_output='/RAM/planner',planner_report_sha=h('planner report'),planner_receipt='/RAM/receipt.json',
        planner_receipt_sha=h('receipt'),planner_source_sha=h('source'),
        consumer_source=str(m.ROOT/'experiments/spectral_reachability'/(m.CONSUMER+'.py')),
        consumer_source_sha=h('consumer source'),consumer_schema=m.CONSUMER+'.consume')
    return m.assemble(args,compiled,consumed,data),data,compiled,consumed


class MemoryStore(OldMemoryStore):
    def check(self,size=4096,final=False):
        m.require(sum(map(len,self.files.values()))+size<=m.CAP-(0 if final else m.FINAL),'RAM own64MiB cap')


def execute(*,fault=None,at=11,close_error=False):
    p,data,_,_=fixture();store=MemoryStore();envs=[]
    def factory():
        env=Env(fault=fault,at=at,close_error=close_error);envs.append(env);return env
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m,'place',fake_place);mp.setattr(m,'validate_env',lambda _:None)
        mp.setattr(m,'sha',lambda path:hashlib.sha256(store.files[Path(path).name]).hexdigest())
        r=m.execute(p,data,store,factory,authorization=m.AUTH,prepared='/RAM/own_prepare.json',prepare_sha256=h('own prepare'))
    return r,store,p,data,envs


@lru_cache(maxsize=1)
def complete():return execute()


def test_actual_repaired_model_plan_abi_and_separate_codecs():
    p,_,_,consumed=fixture();m.validate_execution(p)
    assert m.digest([1,2])!=m.planner_digest([1,2])
    assert [x['root_slot'] for x in consumed['plans']]==[0,1]
    assert all(x['schema']==m.CONSUMER+'.model_plan' for x in consumed['plans'])
    assert m.contract()['fixed_denominator']==2
    assert m.contract()['caps']==dict(constructor=2,close=2,reset=4,public_step=50,physics_step=508,
        set_state=8,goal_setter=6,goal_render=4,goal_bind=2,explicit_render=162)


def test_full_two_raw15_native_lifecycle_and_all162_png():
    r,s,p,_,envs=complete()
    assert r['status']==m.RECORDED,r['errors']
    assert len(m.validate_saved(r,s.events,p))==2
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()}
    assert len([n for n in s.files if n.endswith('.png')])==162
    for row,plan,e in zip(r['rows'],p['plans'],envs,strict=True):
        assert e.closed and e.dispatch_flags==[False]*25 and e.with_target is False
        assert np.asarray(e.commands[:10],np.float32).tobytes()==m.moving.warmup_commands().tobytes()
        assert np.asarray(e.commands[10:],np.float32).tobytes()==m.raw15(plan['commands']).tobytes()
        assert [x['step'] for x in row['checkpoints']]==[0,5,10,15] and row['missing_checkpoints']==[]
        assert row['own_returned']==len(row['responses'])==15 and row['prefix51']['complete']
        assert '_set_state' not in e.__dict__ and 'reset' not in e.__dict__


@pytest.mark.parametrize('field',['order','raw','goal','context','Q','task','warm','terminal','model','digest','partial',
    'missing','oldconsumer','oldplan','blind','pixels','normalizer','logits','codec'])
def test_actual_model_plan_mutations_fail_closed(field):
    p,data,c,u=fixture();x=u['plans'][0]
    if field=='order':u['plans'].reverse()
    elif field=='raw':x['commands'][14][1]=.25
    elif field=='goal':x['goal'][0]+=.1
    elif field=='context':x['original_context']['latent_history'][0][0]+=.1
    elif field=='Q':x['Q'][0][0]=2.
    elif field=='task':x['task_id']=h('bad')
    elif field=='warm':c['slots'][0]['warm_raw10'][0][0]+=.001
    elif field=='terminal':x['terminal']['latent_history'].pop()
    elif field=='model':x['model_sha256']=h('other model')
    elif field=='digest':x['selection_digest']=h('bad')
    elif field=='partial':u['status']='partial'
    elif field=='missing':u['plans'].pop()
    elif field=='oldconsumer':u['schema']=m.PRODUCER+'.consume'
    elif field=='oldplan':x['schema']=m.PRODUCER+'.model_plan'
    elif field=='blind':x['blind_pointer']['opened']=True
    elif field=='pixels':c['slots'][0]['pixel_sha256'][0]=h('bad pixels')
    elif field=='normalizer':x['normalization']['std'][0]=1.
    elif field=='logits':x['logits'][0]=100.
    else:x['selection_digest']=m.digest({k:v for k,v in x.items() if k!='selection_digest'})
    with pytest.raises(ValueError):m.assemble(p['arguments'],c,u,data)


@pytest.mark.parametrize('fault,at',[('truncate',5),('terminal',11),('truncate',15),('truncate',20),('terminal',25)])
def test_native_stop_both_slots_no_padding(fault,at):
    r,s,p,_,envs=execute(fault=fault,at=at)
    assert r['status']==m.RECORDED,r['errors']
    assert len(m.validate_saved(r,s.events,p))==2
    assert all(len(e.commands)==at and e.closed for e in envs)
    assert all(x['own_returned']==max(at-10,0) for x in r['rows'])
    assert all((15 in x['missing_checkpoints'])==(at<25) for x in r['rows'])


@pytest.mark.parametrize('fault',['throw','guard','substitute','render'])
def test_failed_incidents_retained_and_no_action_repair(fault):
    r,s,p,_,envs=execute(fault=fault,close_error=True)
    assert r['status']=='partial_or_failed' and len(r['rows'])==2 and all(e.closed for e in envs)
    assert all(x['first_error'] for x in r['rows'])
    assert [len(e.commands) for e in envs]==([25,11] if fault=='substitute' else [11,11])
    if fault=='throw':assert len(m.validate_saved(r,s.events,p))==2
    else:
        with pytest.raises(ValueError):m.validate_saved(r,s.events,p)


@pytest.mark.parametrize('field',['unknown','late_raw','cp15','reward','warmgoal','goalbind','renderer','count','arm','missing_return'])
def test_saved_invariants_reject_resealed_mutations(field):
    orig,s,prior,_,_=complete();r=deepcopy(orig);p=deepcopy(prior);events=deepcopy(s.events)
    if field=='unknown':events.insert(-1,dict(kind='unknown_event'))
    elif field=='late_raw':
        e=next(e for e in events if e['kind']=='operation_dispatch_intent' and e['receipt']['resource']=='public_step'
            and e['receipt'].get('phase')=='plan' and e['receipt']['step']==15);e['receipt']['raw_bits']='00'
    elif field=='cp15':r['rows'][0]['checkpoints'].pop();r['rows'][0]['missing_checkpoints']=[15]
    elif field=='reward':
        e=next(e for e in events if e['kind']=='returned_transition' and e['phase']=='plan' and e['step']==15);e['reward']=0.
    elif field=='warmgoal':
        e=next(e for e in events if e['kind']=='returned_transition' and e['phase']=='warmup');e['goal_state']=p['source_roots'][0]['goal']
    elif field=='goalbind':
        e=next(e for e in events if e['kind']=='goal_attached');e['goal'][0]+=.01
    elif field=='renderer':
        e=next(e for e in events if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan' and e['receipt']['step']==15)
        e['pre']['renderer']['render_size']=512;e['post']['renderer']['render_size']=512
    elif field=='count':r['counts']['physics_step']['completed']-=1
    elif field=='arm':r['rows'][0]['arm']='joint'
    else:
        e=next(e for e in events if e['kind']=='returned_transition' and e['phase']=='plan' and e['step']==15);events.remove(e)
    reseal(events,r)
    with pytest.raises((ValueError,KeyError,StopIteration)):m.validate_saved(r,events,p)


def test_full_saved_consume_png_and_no_unknown_artifact(monkeypatch):
    r,s,p,data,_=complete()
    monkeypatch.setattr(m,'saved_artifacts',lambda _:deepcopy(s.files))
    monkeypatch.setattr(m,'verify',lambda *a:(deepcopy(p),data))
    out=m.consume('/RAM/physical',hashlib.sha256(s.files['report.json']).hexdigest())
    assert out['saved_PNG_verified']==162 and out['fixed_denominator']==2 and not out['P2_admission']
    monkeypatch.setattr(m,'saved_artifacts',lambda _:dict(s.files,unknown=b'bad'))
    with pytest.raises(ValueError,match='unknown extra'):m.consume('/RAM/physical',hashlib.sha256(s.files['report.json']).hexdigest())


def test_real_store_journal_disk_codec(tmp_path):
    store=m.Store(str(tmp_path/(m.NAME+'_RAMcodec')))
    store.put('binding.json',dict(x=1));store.emit('first',payload=dict(x=1));store.emit('second',payload=dict(x=2))
    body=(store.path/'journal.jsonl').read_bytes()
    assert body.endswith(b'\n') and b'\n\n' not in body
    events=[m.native.decode(x) for x in body.splitlines()]
    assert len(events)==2 and events[1]['previous_sha256']==events[0]['event_sha256']
    for e in events:assert e['event_sha256']==m.source.digest({k:v for k,v in e.items() if k!='event_sha256'})
    assert m.read(store.path/'binding.json')==dict(x=1)


def test_auth_before_reads_or_factory(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('read'))
    with pytest.raises(ValueError):m.run('/RAM/p',h('p'),'/RAM/o',lambda:pytest.fail('factory'),authorization='bad')


def test_producer_source_before_import(monkeypatch):
    p,*_=fixture();monkeypatch.setattr(m,'sha',lambda _:h('bad'))
    monkeypatch.setattr(m.importlib,'import_module',lambda _:pytest.fail('unsealed import'))
    with pytest.raises(ValueError,match='source seal'):m.material(p['arguments'])


def test_signed_zero_action_bits_and_no_clip():
    a=np.zeros((15,2),np.float32);a[0,0]=-0.
    assert m.raw15(a.tolist()).tobytes()==a.tobytes()
    for bad in (a[:10].tolist(),np.full((15,2),1.01).tolist(),[[True,0.]]*15):
        with pytest.raises(ValueError):m.raw15(bad)


@pytest.mark.parametrize('mutation',['source','schema','receiptSHA','receiptBody','receiptSignedZero','consumerBundle','fixPolicy'])
def test_new_consumer_authority_seals_fail_before_source_material(monkeypatch,tmp_path,mutation):
    p,data,c,u=fixture();args=deepcopy(p['arguments'])
    output=tmp_path/'actual_saved_model';output.mkdir()
    args.update(planner_output=str(output),planner_receipt=str(tmp_path/'actual_receipt.json'))
    consumer_path=Path(args['consumer_source']);producer_path=m.ROOT/'experiments/spectral_reachability'/(m.PRODUCER+'.py')
    u['consumer_bindings']={str(consumer_path):args['consumer_source_sha']}
    report=dict(prepared=str(tmp_path/'actual_prepare.json'),prepare_sha256=u['prepare_sha256'])
    def sh(path):
        path=Path(path)
        if path==producer_path:return args['planner_source_sha']
        if path==consumer_path:return h('wrong consumer') if mutation=='source' else args['consumer_source_sha']
        if path==output/'report.json':return args['planner_report_sha']
        if path==Path(args['planner_receipt']):return h('bad receipt') if mutation=='receiptSHA' else args['planner_receipt_sha']
        return h(str(path))
    def read(path,*a):
        if Path(path)==output/'report.json':return deepcopy(report)
        value=deepcopy(u)
        if mutation=='receiptBody':value['errors']=['altered']
        if mutation=='receiptSignedZero':value['plans'][0]['commands'][0][0]=-0.
        return value
    if mutation=='schema':args['consumer_schema']=m.PRODUCER+'.consume'
    if mutation=='consumerBundle':u['consumer_bindings'][str(consumer_path)]=h('bad self seal')
    if mutation=='fixPolicy':u['fix_policy']='normalize all signed zeros'
    producer=SimpleNamespace(__file__=str(producer_path),verify=lambda *a:deepcopy(c))
    consumer=SimpleNamespace(__file__=str(consumer_path),consume=lambda *a:deepcopy(u))
    monkeypatch.setattr(m,'sha',sh);monkeypatch.setattr(m,'read',read)
    monkeypatch.setattr(m.importlib,'import_module',lambda name:producer if name.endswith(m.PRODUCER) else consumer)
    if mutation=='fixPolicy':
        # Reach assemble only after whole receipt matches; still zero actual source/NN/physics.
        monkeypatch.setattr(m.native,'source_material',lambda:deepcopy(data))
    else:monkeypatch.setattr(m.native,'source_material',lambda:pytest.fail('must reject authority before source replay inputs'))
    with pytest.raises(ValueError):m.material(args)
