"""Synthetic RAM/native-IO callbacks and tmp codec only; no simulator or NN."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import agent_p3_h3_witness_physical_20261029_v1 as m
from experiments.spectral_reachability import agent_p3_shared_panel_20261029_v1 as shared
from test_agent_allmode_physical_20261028_v1 import fixture as source_fixture, MemoryStore, Env, fake_place, h, reseal
import agent_p3_fullcommit_live_20261028_v1_tests as helpers


@lru_cache(maxsize=1)
def _fixture():
    _,data,*_=source_fixture();meta=m.native.allmode.root_metadata(data)
    compiled=dict(schema=m.PRODUCER+'.prepare',contract=shared.contract(),tasks=[],layers=[])
    for slot in (0,1):
        base,_=helpers.request(slot=slot)
        base['binding']['root_sha256']=meta[slot]['root_sha256']
        compiled['tasks'].append(base)
        compiled['layers'].append(dict(source_slot=slot,commands=np.zeros((10,2),np.float32).tolist(),
            task_id=h(('task',slot)),selection_sha256=h(('selection',slot))))
    report=dict(schema=m.PRODUCER+'.report',contract=shared.contract(),prepared='/RAM/shared_prepare.json',
        prepare_sha256=m.PREPARE_SHA,nodes=shared.node_roster(),rows=shared.native_roster())
    for i,(slot,_,bank) in enumerate(m.ORDER):
        mode='adaptive_'+bank;eid=f's{slot}_{mode}';root=data['roots'][slot]
        e=Env();e.reset(seed=root['slot']['reset_seed'])
        from test_p1_constructive_moving_history_physical_20261027_v1 import REQUEST
        obs,_=e.reset(seed=0,options=dict(state=REQUEST.copy(),goal_state=e.goal_state.copy()))
        for a in m.moving.warmup_commands():obs,*_=e.step(a.copy())
        m.moving.attach_goal(e,root['goal'],root['packets']['off']['goal_rgb'])
        prefix=shared.bridge.prefix_binding(compiled['layers'][slot])
        for a in np.asarray(prefix['commands'],np.float32):obs,*_=e.step(a.copy())
        image=e.render();buf=io.BytesIO();Image.fromarray(image).save(buf,format='PNG')
        cp=dict(phase='plan',step=5,state=m.source.vector(obs['state']).tolist(),snapshot=m.invariant(e)['snapshot'],
            pixel_sha256=m.moving.image_sha(image),file_sha256=hashlib.sha256(buf.getvalue()).hexdigest(),
            goal=meta[slot]['goal'],goal_pose=meta[slot]['goal_pose'],goal_rgb_sha256=meta[slot]['goal_pixels']['off'])
        base=compiled['tasks'][slot];raw10=m.moving.warmup_commands()
        context=deepcopy(base['root_context']);context['action_history']=shared.moving.normalized(base,raw10)
        ob=dict(boundary=5,png_sha256=cp['file_sha256'],pixel_sha256=cp['pixel_sha256'],encoding_sha256=h('encoding'),
            observed_index=1,encoded_index=2,observed_latent_sha256=shared.old.packet_digest(context['latent_history'][-1]))
        req=shared.make_request(base,context,slot=slot,arm=mode,boundary=5,raw10=raw10.tolist(),observation=ob)
        # Literal fixture arrays, NOT model-generated actions. Distinct tail from FIRST5.
        actions=np.zeros((15,2),np.float32);actions[5:]=np.float32(.125*(i+1))
        candidates=[dict(index=j,status='returned',actions=actions.tolist(),whole=[{'synthetic':j}]*3,
            segmented=[{'synthetic':j}]*3,cost=float(4-j)) for j in range(4)]
        winner=deepcopy(candidates[3])
        panels=[dict(horizon=1),dict(horizon=3,status='returned',winner=winner,candidates=candidates,selected_path=winner['whole'])]
        res=dict(status='sealed',selected_horizon=3,request_sha256=shared.digest(req),panels=panels,commands=actions[:5].tolist())
        n=next(n for n in report['nodes'] if n['episode_id']==eid and n['boundary']==5)
        n.update(status='sealed',request=req,receipt=res)
        row=next(r for r in report['rows'] if r['episode_id']==eid)
        row.update(prefix_exact=True,initial_exact=True,prefix=prefix,checkpoints=[cp])
    args=dict(planner_output='/RAM/shared',planner_report_sha=m.REPORT_SHA,planner_receipt='/RAM/strict.json',
        planner_receipt_sha=m.RECEIPT_SHA,planner_source_sha=h('source'))
    p=m.assemble(args,compiled,report,data)
    return p,data,compiled,report


def fixture():return deepcopy(_fixture())


def execute(*,fault=None,at=24,close_error=False,damage_cp=False):
    p,data,_,_=fixture();s=MemoryStore();envs=[]
    if damage_cp:
        w=p['plans'][0]['source_witness'];w['actual_cp5']['state'][0]+=.25
        w['witness_sha256']=m.digest({k:v for k,v in w.items() if k!='witness_sha256'})
        p['plans'][0]['selection_digest']=m.digest({k:v for k,v in p['plans'][0].items() if k!='selection_digest'})
    def factory():
        e=Env(fault=fault,at=at,close_error=close_error);envs.append(e);return e
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m,'place',fake_place);mp.setattr(m,'validate_env',lambda _:None)
        mp.setattr(m,'sha',lambda path:hashlib.sha256(s.files[Path(path).name]).hexdigest())
        r=m.execute(p,data,s,factory,authorization=m.AUTH,prepared='/RAM/own_prepare.json',prepare_sha256=h('prepare'))
    return r,s,p,data,envs


@lru_cache(maxsize=1)
def complete():return execute()


def test_complete4_exact_prefix5_full15_native_census():
    r,s,p,data,envs=complete()
    assert r['status']==m.RECORDED,r['errors']
    assert len(m.validate_saved(r,s.events,p))==4
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()}
    assert len([f for f in s.files if f.endswith('.png')])==384
    for row,plan,e in zip(r['rows'],p['plans'],envs,strict=True):
        assert e.closed and len(e.commands)==30 and e.dispatch_flags==[False]*30
        assert np.asarray(e.commands[10:],np.float32).tobytes()==m.raw20(plan['commands']).tobytes()
        assert np.asarray(e.commands[15:],np.float32).tobytes()==m.raw15(plan['suffix_commands']).tobytes()
        assert [c['step'] for c in row['checkpoints']]==list(m.CP) and not row['missing_checkpoints']
        assert row['own_returned']==20 and row['prefix51']['complete']
        assert not r['C3'] and not r['P3_admission'] and r['new_neural_calls']==0


@pytest.mark.parametrize('field',('horizon','first5','winner','prefix','request','root','goal','cp5','missing','duplicate'))
def test_saved_source_witness_abi_refuses_mutations(field):
    p,data,c,r=fixture();n=next(n for n in r['nodes'] if n['episode_id']=='s0_adaptive_endpoints' and n['boundary']==5)
    row=next(x for x in r['rows'] if x['episode_id']==n['episode_id'])
    if field=='horizon':n['receipt']['selected_horizon']=1
    elif field=='first5':n['receipt']['commands'][0][0]=.25
    elif field=='winner':n['receipt']['panels'][1]['winner']['actions'][14][0]=.25
    elif field=='prefix':row['prefix']['commands'][0][0]=.25
    elif field=='request':n['request']['task']['goal'][0]+=.25
    elif field=='root':c['tasks'][0]['binding']['root_sha256']=h('wrong root')
    elif field=='goal':row['checkpoints'][0]['goal'][0]+=.25
    elif field=='cp5':row['checkpoints'][0]['pixel_sha256']=h('bad png')
    elif field=='missing':r['nodes'].remove(n)
    else:r['nodes'][-1]=deepcopy(n)
    with pytest.raises((ValueError,KeyError)):m.assemble(p['arguments'],c,r,data)


@pytest.mark.parametrize('at',(5,11,24,29,30))
def test_earlystop_retains_exact_event_time_and_missing_cp20(at):
    r,s,p,_,envs=execute(fault='truncate',at=at)
    assert r['status']==m.RECORDED,r['errors']
    assert len(m.validate_saved(r,s.events,p))==4
    assert all(len(e.commands)==at and e.closed for e in envs)
    assert all(row['own_returned']==max(0,at-10) for row in r['rows'])
    assert all((20 in row['missing_checkpoints'])==(at<30) for row in r['rows'])


def test_cp5_mismatch_never_dispatches_suffix_or_repairs_context():
    r,s,p,_,envs=execute(damage_cp=True)
    assert r['status']=='partial_or_failed'
    assert len(envs[0].commands)==15 and envs[0].closed
    assert r['rows'][0]['first_error'] and all(len(e.commands)==30 for e in envs[1:])
    with pytest.raises(ValueError,match='cp5'):m.validate_saved(r,s.events,p)


def test_failed_attempts_not_refunded_all4_preserved():
    r,s,p,_,envs=execute(fault='throw',at=21,close_error=True)
    assert r['status']=='partial_or_failed' and len(r['rows'])==4 and all(e.closed for e in envs)
    assert r['counts']['public_step']['attempted']==84 and r['counts']['public_step']['completed']==80
    assert all(x['own_attempted']==11 and x['own_returned']==10 and x['first_error'] for x in r['rows'])
    assert len(m.validate_saved(r,s.events,p))==4


@pytest.mark.parametrize('damage',('unknown','raw20','cp20','reward','native7','goal','count','return'))
def test_resealed_saved_mutation_rejected(damage):
    r0,s,p0,_,_=complete();r=deepcopy(r0);p=deepcopy(p0);es=deepcopy(s.events)
    if damage=='unknown':es.insert(-1,dict(kind='unknown'))
    elif damage=='raw20':
        e=next(e for e in es if e['kind']=='operation_dispatch_intent' and e['receipt']['resource']=='public_step'
            and e['receipt'].get('phase')=='plan' and e['receipt']['step']==20);e['receipt']['raw_bits']='00'
    elif damage=='cp20':r['rows'][0]['checkpoints'].pop()
    elif damage=='count':r['counts']['physics_step']['completed']-=1
    else:
        e=next(e for e in es if e['kind']=='returned_transition' and e['phase']=='plan' and e['step']==20)
        if damage=='return':es.remove(e)
        elif damage=='reward':e['reward']=0.
        elif damage=='native7':e['state'][5]+=.25
        else:e['goal_state'][0]+=.25
    reseal(es,r)
    with pytest.raises((ValueError,KeyError,StopIteration)):m.validate_saved(r,es,p)


def test_full_saved_consume_all384_png_no_unknown_extra(monkeypatch):
    r,s,p,data,_=complete()
    monkeypatch.setattr(m,'saved_artifacts',lambda _:deepcopy(s.files))
    monkeypatch.setattr(m,'verify',lambda *a:(deepcopy(p),data))
    out=m.consume('/RAM/replay',hashlib.sha256(s.files['report.json']).hexdigest())
    assert out['saved_PNG_verified']==384 and len(out['plans'])==len(out['rows'])==4
    assert out['all_four_physically_executed'] and not out['C3']
    monkeypatch.setattr(m,'saved_artifacts',lambda _:dict(s.files,unknown=b'x'))
    with pytest.raises(ValueError,match='unknown extra'):m.consume('/RAM/replay',hashlib.sha256(s.files['report.json']).hexdigest())


def test_real_store_disk_emit_put_and_native_hashchain(tmp_path):
    s=m.Store(str(tmp_path/(m.NAME+'_codec')));s.put('binding.json',dict(x=1))
    s.emit('first',value=1);s.emit('second',value=2)
    b=(s.path/'journal.jsonl').read_bytes()
    assert b.endswith(b'\n') and b'\n\n' not in b
    es=[m.native.decode(line) for line in b.splitlines()]
    assert len(es)==2 and es[0]['previous_sha256'] is None and es[1]['previous_sha256']==es[0]['event_sha256']
    for e in es:assert e['event_sha256']==m.source.digest({k:v for k,v in e.items() if k!='event_sha256'})
    assert m.read(s.path/'binding.json')==dict(x=1)
    with pytest.raises(FileExistsError):s.put('binding.json',dict(x=2))


def test_future_native_outcomes_cannot_change_fixed_witness_extraction():
    _,_,c,r=fixture();before=m.select_witnesses(c,r)
    for row in r['rows']:
        row.update(native_success=True,final_metrics={'unrelated_future_outcome':123},final_state=[123.]*7)
    assert m.select_witnesses(c,r)==before


def test_auth_before_input_and_factory(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('unauthorized read'))
    with pytest.raises(ValueError):m.run('/RAM/p',h('p'),'/RAM/o',lambda:pytest.fail('factory'),authorization='bad')


def test_source_hash_before_import_or_fullstrict(monkeypatch):
    p,*_=fixture();monkeypatch.setattr(m,'sha',lambda _:h('wrong code'))
    monkeypatch.setattr(m.importlib,'import_module',lambda _:pytest.fail('early import/strict'))
    with pytest.raises(ValueError,match='source before'):m.material(p['arguments'])


@pytest.mark.parametrize('n',(5,15,20))
def test_raw_float32_signedzero_no_projection_or_bool(n):
    x=np.zeros((n,2),np.float32);x[0,0]=-0.
    assert m.literal(x.tolist(),n).tobytes()==x.tobytes()
    for bad in ([[True,0.]]*n,np.full((n,2),1.00001).tolist(),np.full((n,2),float('nan')).tolist()):
        with pytest.raises(ValueError):m.literal(bad,n)
