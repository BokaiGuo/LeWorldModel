"""RAM arrays/real JSON layout and virtual files, never native physics/neural."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import time

import numpy as np
import pytest

from experiments.spectral_reachability import agent_moving_CEM8_physical_20261028_v1 as m
from experiments.spectral_reachability import agent_moving_CEM8_20261028_v1 as producer
import test_capability_suffix_factorial_physical_20261027_v1 as old
from test_p1_moving_overlay_recompile_physical_20261027_v1 import Env, fake_place, h


class MemoryStore(old.MemoryStore):
    def check(self,size=4096,final=False):
        m.require(self.used+size <= m.CAP-(0 if final else m.FINAL) and self.free >= m.FLOOR+size, 'RAM actual cap/floor')


def layout():
    original,data,images = deepcopy(old.frozen_fixture()); branches,rows,plans = [],[],[]
    for i,p in enumerate(original['plans']):
        b = dict(anchor=i,source_slot=p['root_slot'],condition=p['condition'],arm=p['arm'],parent_task_id=p['task_id'],
                 blind_sha256=p['blind_sha256'],planner_selection_sha256=p['selection_digest'],original_plan=deepcopy(p),
                 observed_context=dict(latent_history=[[float(i)]*192]*3,action_history=[[.1]*10]*2),
                 prefix_commands=deepcopy(p['commands'][:5]))
        b['branch_id'] = m.core.digest(b)
        suffix = np.full((5,2), np.nextafter(np.float32(.2),np.float32(1)), np.float32).tolist()
        row = dict(index=i,status='suffix_sealed',selected_index=3,suffix_commands=suffix,model_Q=.25+i,replay_exact=True,
                   evaluations=[dict(commands=deepcopy(suffix),cost=.25+i) for _ in range(65)])
        plan = producer.model_plan(b,row); row['plan'] = deepcopy(plan)
        plans.append(dict(plan,whole_run_report_sha256=h('CEM actual report'))); branches.append(b); rows.append(row)
    # This models the actual JSON roundtrip, not shared mutable Python references.
    pp = json.loads(m.raw(dict(schema=producer.NAME+'.prepare',contract=producer.contract(),branches=branches,bindings={},inherited_prepare_sha256=h('parent'))))
    c = json.loads(m.raw(dict(schema=producer.NAME+'.report',status='teacher8_CEM_commands_sealed_not_admitted',errors=[],first_error=None,
              commands_released=True,model_unchanged=True,source_verified=True,P1_admission=False,P3_admission=False,
              contract=producer.contract(),rows=rows,resources={k:dict(attempted=520,completed=520) for k in ('predict','action_encoder','pred_proj')},
              prepared='/RAM/CEM_prepare.json',prepare_sha256=h('CEM prepare'),artifact_sha256={})))
    v = json.loads(m.raw(dict(schema=producer.NAME+'.consume',status='teacher8_CEM_commands_verified_not_admitted',errors=[],plans=plans,
                            P1_admission=False,P3_admission=False,equal_total_compute=False,new_neural_calls=0,new_physics_calls=0)))
    return c,pp,v,original,data,images


def fixture():
    c,pp,v,original,data,images = layout()
    p = dict(schema=m.NAME+'.prepare',contract=m.contract(),plans=original['plans'],anchors=original['anchors'],
             suffix_plans=m.validate_verified(c,pp,v,original['plans'],h('CEM actual report')),MAIN_verified8=True,bindings={},
             compiler_report_sha256=h('CEM actual report'),consumer_report_sha256=h('MAIN accepted'),
             physical_report_sha256=m.ORIGINAL_SHA,planner_report_sha256=original['planner_report_sha256'])
    return p,data,images


def execute(monkeypatch,*,p=None,factory=Env,started=None,store=None):
    fp,data,images = fixture(); p = fp if p is None else p; store = MemoryStore() if store is None else store
    store.free = m.FLOOR+m.CAP; envs=[]
    def make():
        e=factory(); envs.append(e); return e
    monkeypatch.setattr(m.base,'place',fake_place); monkeypatch.setattr(m.base,'validate_env',lambda _:None)
    monkeypatch.setattr(m.base,'old_image',lambda ref:images[ref['path']].copy())
    monkeypatch.setattr(m,'sha',lambda p:hashlib.sha256(store.files[Path(p).name]).hexdigest())
    def forbidden(*a,**k): raise AssertionError('old executor/probe/witness/endpoint forbidden')
    monkeypatch.setattr(m.replay,'execute',forbidden); monkeypatch.setattr(m.probe,'execute',forbidden)
    monkeypatch.setattr(m.base,'execute',forbidden); monkeypatch.setattr(m.base,'episode',forbidden)
    monkeypatch.setattr(m.base.moving,'witness',forbidden)
    r=m.execute(p,data,store,make,authorization=m.AUTH,prepared='/RAM/physical_prepare.json',prepare_sha256=h('physical prepare'),started=started)
    return r,store,envs,p


@lru_cache(maxsize=1)
def completed():
    with pytest.MonkeyPatch.context() as monkeypatch:
        return execute(monkeypatch)


def test_exact8_original51_then_own_CEM_literal_raw_and_all_resources():
    r,s,envs,p=completed()
    assert r['status']==m.RECORDED,r['errors']
    assert len(r['rows'])==len(envs)==r['fixed_denominator']==r['full_suffixes']==8
    assert r['known_roots']==2 and r['fresh_roots']==0 and not r['independent_confirmation']
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()}
    assert len([x for x in s.files if x.endswith('.png')])==528
    assert len(m.validate_saved(r,s.events,p))==8
    for i,(row,env) in enumerate(zip(r['rows'],envs,strict=True)):
        assert env.closed and env.dispatch_flags==[False]*20 and len(env.commands)==20
        assert np.asarray(env.commands[10:15],np.float32).tobytes()==m.raw5(p['suffix_plans'][i]['plan']['prefix_commands']).tobytes()
        assert np.asarray(env.commands[15:],np.float32).tobytes()==m.raw5(p['suffix_plans'][i]['plan']['suffix_commands']).tobytes()
        assert len(row['final_state'])==7 and [c['step'] for c in row['checkpoints']]==[0,5,10]
        assert row['checkpoints'][-1]['phase']=='suffix' and row['missing_checkpoints']==[]
        assert env.goal_pose.tolist()==[300,300,0] and '_set_state' not in env.__dict__ and 'reset' not in env.__dict__


@pytest.mark.parametrize('field',['partial','consumer','missing','order','anchor','bool','root','condition','prefixarm','task','blind','parent','context','originalplan',
                                 'prefix','suffixULP','replay','Q','report','seal','resources','selected'])
def test_real_producer_ABI_mutations_rejected(field):
    c,pp,v,o,_,_=layout();p=v['plans'][0]
    if field=='partial':c['status']='partial_or_failed'
    elif field=='consumer':v['status']='teacher8_CEM_commands_sealed_not_admitted'
    elif field=='missing':v['plans'].pop()
    elif field=='order':v['plans'][0],v['plans'][1]=v['plans'][1],v['plans'][0]
    elif field=='anchor':p['anchor']=1
    elif field=='bool':p['root_slot']=False
    elif field=='root':p['source_slot']=1
    elif field=='condition':p['condition']='on'
    elif field=='prefixarm':p['prefix_arm']=m.base.ARMS[1]
    elif field=='task':p['task_id']=h('wrong')
    elif field=='blind':p['blind_sha256']=h('wrong')
    elif field=='parent':p['original_selection_digest']=h('wrong')
    elif field=='context':p['teacher_context_sha256']=h('wrong')
    elif field=='originalplan':pp['branches'][0]['original_plan']['source_report_sha256']=h('wrong')
    elif field=='prefix':p['prefix_commands'][0][0]=0.
    elif field=='suffixULP':p['suffix_commands'][0][0]=float(np.nextafter(np.float32(p['suffix_commands'][0][0]),np.float32(1)))
    elif field=='replay':c['rows'][0]['evaluations'][64]['commands'][0][0]=0.
    elif field=='Q':p['model_Q']=0.
    elif field=='report':p['whole_run_report_sha256']=h('wrong')
    elif field=='seal':p['selection_sha256']=h('wrong')
    elif field=='resources':c['resources']['predict']['completed']=519
    else:p['selected_index']=64
    if field!='seal':p['selection_sha256']=m.core.digest({k:x for k,x in p.items() if k not in ('selection_sha256','whole_run_report_sha256')})
    with pytest.raises((ValueError,KeyError)):m.validate_verified(c,pp,v,o['plans'],h('CEM actual report'))


@pytest.mark.parametrize('field',['state','snapshot','nonselected_on','goal_pose','file_sha256'])
def test_any51_mismatch_blocks_only_affected_anchor_before_suffix(monkeypatch,field):
    p,_,_=fixture();a=p['anchors'][0]
    if field=='state':a['checkpoint']['state'][0]+=1e-9
    elif field=='snapshot':a['checkpoint']['snapshot']['bodies']['block']['angular_velocity']+=1e-9
    elif field=='nonselected_on':a['trace'][1]['receipt']['pixel_sha256']=h('wrong')
    elif field=='goal_pose':a['trace'][0]['pre']['goal_pose'][0]+=1e-9
    else:a['checkpoint']['file_sha256']=h('wrong')
    r,s,envs,_=execute(monkeypatch,p=p)
    assert r['status']=='partial_or_failed' and len(r['rows'])==8
    assert len(envs[0].commands)==15 and r['rows'][0]['suffix_attempted']==0 and not r['rows'][0]['start_verified']
    assert all(x['status']=='suffix_completed' for x in r['rows'][1:]) and all(e.closed for e in envs)
    assert 'b0_plan_05_0.png' in s.files


@pytest.mark.parametrize('fault',['throw','substitute','truncate','terminal'])
def test_native_stop_partial_no_padding_no_refund_cleanup(monkeypatch,fault):
    r,s,envs,p=execute(monkeypatch,factory=lambda:Env(fault=fault,at=16))
    assert len(r['rows'])==8 and all(e.closed and len(e.commands)==16 for e in envs)
    assert all(x['suffix_attempted']==1 and x['suffix_returned']==(0 if fault=='throw' else 1) for x in r['rows'])
    assert all(x['missing_checkpoints']==[10] for x in r['rows'])
    if fault in ('truncate','terminal'):
        assert r['status']==m.RECORDED and r['early_stops']==8 and len(m.validate_saved(r,s.events,p))==8
    else:assert r['status']=='partial_or_failed' and r['engineering_failures']==8
    if fault=='throw':assert r['counts']['public_step']==dict(attempted=128,completed=120)


def test_expired_before_factory_keeps8_unknown(monkeypatch):
    r,_,envs,_=execute(monkeypatch,started=time.monotonic()-901)
    assert not envs and r['counts']['constructor']['attempted']==0 and len(r['rows'])==8
    assert all(x['final_state'] is x['start_state'] is x['native_success'] is None for x in r['rows'])


def test_partial_constructor_internal_work_unknown(monkeypatch):
    def fail():raise RuntimeError('constructor failed internally')
    r,_,envs,_=execute(monkeypatch,factory=fail)
    assert not envs and r['counts']['constructor']==dict(attempted=8,completed=0)
    assert r['total_internal_physics_scope']=='unknown_constructor_work'


def test_meter_actual_return_journal_failure_and_close_cleanup():
    s=MemoryStore();s.free=m.FLOOR+m.CAP;meter=m.Meter(s);emit=s.emit;calls=[]
    def fault(kind,**kw):
        if kind=='operation_returned':raise OSError('return journal')
        emit(kind,**kw)
    s.emit=fault
    with pytest.raises(OSError):meter.call('public_step',lambda:calls.append('returned'))
    assert meter.counts['public_step']==dict(attempted=1,completed=1) and calls==['returned']
    meter.call('close',lambda:calls.append('closed'),cleanup=True)
    assert calls==['returned','closed'] and meter.secondary


def test_meter_cap_failure_floor_and_authorization(monkeypatch):
    s=MemoryStore();s.free=m.FLOOR+m.CAP;meter=m.Meter(s)
    with pytest.raises(RuntimeError):meter.call('public_step',lambda:(_ for _ in ()).throw(RuntimeError('attempt failed')))
    assert meter.counts['public_step']==dict(attempted=1,completed=0)
    meter.counts['constructor']['attempted']=8
    with pytest.raises(ValueError):meter.call('constructor',lambda:pytest.fail('cap'))
    s.free=m.FLOOR-1
    with pytest.raises(ValueError):meter.call('public_step',lambda:pytest.fail('floor'))
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('unauthorized read'))
    with pytest.raises(ValueError):m.run('x',h('x'),'/RAM/x',lambda:pytest.fail('factory'),authorization='wrong')


def reseal(events,report):
    tail=None
    for i,e in enumerate(events):
        e.update(index=i,previous_sha256=tail);e['event_sha256']=m.base.source.digest({k:v for k,v in e.items() if k!='event_sha256'});tail=e['event_sha256']
    report['journal_chain_tail']=tail


@pytest.mark.parametrize('field',['count','row','prefix','onframe','goal','response','delta','metric','terminal','checkpoint','scope','hashchain','order','denominator','unknown'])
def test_saved_validator_mutations_even_with_resealed_chain(field):
    r0,s,_,p0=completed();r,p,events=deepcopy(r0),deepcopy(p0),deepcopy(s.events)
    if field=='count':r['counts']['physics_step']['completed']-=1
    elif field=='row':r['rows'][0]['selected_suffix']['plan']['suffix_commands'][0][0]=0.
    elif field=='prefix':next(e for e in events if e['kind']=='operation_dispatch_intent' and e['receipt']['resource']=='public_step' and e['receipt'].get('phase')=='plan')['receipt']['raw_bits']='bad'
    elif field=='onframe':next(e for e in events if e['kind']=='fresh_observation' and e['receipt']['overlay_index']==1)['receipt']['pixel_sha256']=h('wrong')
    elif field=='goal':next(e for e in events if e['kind']=='fresh_observation')['pre']['goal_pose'][0]+=1
    elif field=='response':r['rows'][0]['final_state'][0]+=1
    elif field=='delta':r['rows'][0]['responses'][0]['delta_native7'][0]+=1
    elif field=='metric':r['rows'][0]['responses'][0]['native_metrics']['native_distance']=0.
    elif field=='terminal':r['rows'][0]['responses'][0]['terminated']=True
    elif field=='checkpoint':r['rows'][0]['checkpoints'][-1]['state'][0]+=1
    elif field=='scope':next(e for e in events if e.get('branch')==0 and e['kind']=='fresh_observation')['prefix_arm']='other'
    elif field=='hashchain':events[3]['event_sha256']=h('bad')
    elif field=='order':r['rows'][0],r['rows'][1]=r['rows'][1],r['rows'][0]
    elif field=='denominator':r['rows'].pop()
    else:r['rows'][0]['native_success']=None
    if field!='hashchain':reseal(events,r)
    with pytest.raises((ValueError,KeyError)):m.validate_saved(r,events,p)


def arguments():
    return dict(compiler_output='/RAM/CEM',compiler_report_sha256=h('CEM actual report'),producer_source_sha256=m.PRODUCER_SHA,
                consumer_source_sha256=m.RELEASE_SHA,consumer_report='/RAM/release/receipt.json',consumer_report_sha256=h('accepted'),
                physical_output='/RAM/original8',physical_report_sha256=m.ORIGINAL_SHA)


def compiler_fixture(monkeypatch,mutation=None):
    c,pp,v,_,_,_=layout();args=arguments();parent='/RAM/parent.json';parent_report='/RAM/parent/report.json'
    saved=dict(schema=m.RELEASE+'.receipt',status='strict_CEM8_release_persisted_not_admitted',producer_result=deepcopy(v),
               arguments=dict(directory='/RAM/CEM',report_sha=args['compiler_report_sha256'],prepared=c['prepared'],prepare_sha=c['prepare_sha256']),
               input_count=1,input_sha256={'/RAM/complete_release_source.py':h('release source')},request_sha256=h('request'))
    pp['bindings']={parent:h('parent'),parent_report:h('parent report')}
    sources={str(m.ROOT/'experiments/spectral_reachability'/(m.PRODUCER+'.py')):m.PRODUCER_SHA,
             str(m.ROOT/'experiments/spectral_reachability'/(m.RELEASE+'.py')):m.RELEASE_SHA,
             '/RAM/CEM/report.json':args['compiler_report_sha256'],'/RAM/release/receipt.json':args['consumer_report_sha256'],
             '/RAM/CEM_prepare.json':c['prepare_sha256'],parent:h('parent'),parent_report:h('parent report')}
    parent_p=dict(bindings={'/RAM/full_source.py':h('full source')});parent_r=dict(artifact_sha256={'array.npy':h('parent array')})
    if mutation=='source':sources[str(m.ROOT/'experiments/spectral_reachability'/(m.PRODUCER+'.py'))]=h('wrong')
    elif mutation=='report':sources['/RAM/CEM/report.json']=h('wrong')
    elif mutation=='receipt':sources['/RAM/release/receipt.json']=h('wrong')
    elif mutation=='raw':saved['status']='teacher8_CEM_commands_sealed_not_admitted'
    elif mutation=='mismatch':saved['producer_result']['plans'][0]['model_Q']+=1
    elif mutation=='inventory':saved['input_count']=2
    elif mutation=='invocation':saved['arguments']['report_sha']=h('wrong')
    elif mutation=='wrapper':sources[str(m.ROOT/'experiments/spectral_reachability'/(m.RELEASE+'.py'))]=h('wrong')
    elif mutation=='zero':args['consumer_report_sha256']='0'*64
    elif mutation=='missing':del args['consumer_source_sha256']
    elif mutation=='absent':sources.pop('/RAM/CEM/report.json')
    calls=[]
    def verify(*a):
        calls.append(a)
        m.require(saved['producer_result']==v,'actual release verifier checks complete own result')
        return dict(status='release_reverified_not_admitted',plans=8,sha256=args['consumer_report_sha256'])
    module=NS(__file__=str(m.ROOT/'experiments/spectral_reachability'/(m.PRODUCER+'.py')),
              FACTORIAL_PREP=Path(parent),FACTORIAL_OUTPUT=Path('/RAM/parent'))
    release=NS(__file__=str(m.ROOT/'experiments/spectral_reachability'/(m.RELEASE+'.py')),verify=verify)
    monkeypatch.setattr(m,'sha',lambda p:sources[str(p)])
    monkeypatch.setattr(m,'read',lambda p,*a:{'/RAM/release/receipt.json':saved,'/RAM/CEM/report.json':c,'/RAM/CEM_prepare.json':pp,parent:parent_p,parent_report:parent_r}[str(p)])
    monkeypatch.setattr(m.importlib,'import_module',lambda name:{'experiments.spectral_reachability.'+m.PRODUCER:module,
                                                              'experiments.spectral_reachability.'+m.RELEASE:release}[name])
    return args,calls


def test_external_actual_consume_and_full_parent_closure(monkeypatch):
    args,calls=compiler_fixture(monkeypatch);c,pp,v,bindings=m.compiler_material(args)
    assert calls==[('/RAM/release',args['consumer_report_sha256'])]
    assert '/RAM/full_source.py' in bindings and '/RAM/parent/array.npy' in bindings and '/RAM/complete_release_source.py' in bindings
    assert v['schema']==m.PRODUCER+'.consume' and '/RAM/release/request.json' in bindings


@pytest.mark.parametrize('mutation',['source','report','receipt','raw','mismatch','zero','missing','absent','inventory','invocation','wrapper'])
def test_prepare_authority_failclosed_before_actual_output_exists(monkeypatch,mutation):
    args,calls=compiler_fixture(monkeypatch,mutation)
    with pytest.raises((ValueError,KeyError)):m.compiler_material(args)
    if mutation not in ('mismatch','inventory'):assert not calls


def test_prepare_size_regression_preserves_full_binding_table():
    p,_,_=fixture()
    # Real parent producer contract describes6162 inherited entries. Use long
    # absolute names, all source hashes, and the entire8x51 trace without thinning.
    p['bindings']={str(m.ROOT/f'results/full_inherited_native_modules_20261028/branch_{i:05d}.predictor.input_output.receipt.npy.gz'):h(i) for i in range(9000)}
    payload=m.raw(p)
    assert len(p['bindings'])==9000 and len(p['anchors'])==8 and all(len(a['trace'])==51 for a in p['anchors'])
    assert 512*1024 < len(payload) < m.PREPARE_CAP and m.PREPARE_CAP==16*1024**2


def test_RAM_complete_output_report_size_regression():
    r,s,_,p=completed()
    assert s.used < m.CAP and len(m.raw(r)) < m.FINAL and len(m.raw(p)) < m.PREPARE_CAP
