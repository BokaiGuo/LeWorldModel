"""Pure RAM numeric transitions, not native physics/neural; no tmp/cache payloads."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import moving_history_observed_suffix_physical_20261027_v1 as m
import test_p1_constructive_moving_history_physical_20261027_v1 as oldtests

MemoryStore=oldtests.MemoryStore


@lru_cache(maxsize=1)
def original_fixture():
    p,d=oldtests.fixture();p['schema']=m.base.NAME+'.prepare'
    for n,row in enumerate(p['plans']):row['arm']=m.ARMS[n%2]
    report,store,_=oldtests.execute(p=p,data=d)
    assert report['status']=='moving_history_physical_recorded'
    paired=m.assemble(report,p,store.events)
    return p,d,report,store,paired


def fixture():
    oldp,oldd,r,_,paired=original_fixture();d=deepcopy(oldd)
    d.update(paired=deepcopy(paired),original_prepared=deepcopy(oldp),original_report=deepcopy(r))
    plans=deepcopy(oldp['plans'])
    for n,p in enumerate(plans):
        p['commands'][5:]=np.full((5,2),.125 if n%2==0 else -.125,np.float32).tolist()
        p['selection_digest']='new_'+str(n)
    p=dict(schema=m.NAME+'.prepare',plans=plans,contract=m.contract(60),bindings={},sourcepack_sha256=oldp['sourcepack_sha256'])
    return p,d


class ArrayEnv(oldtests.ArrayEnv):
    def __init__(self,*,extra=None,at=13,**kw):super().__init__(at=at,**kw);self.extra=extra;self.extra_at=at
    def render(self):
        a=super().render()
        if len(self.commands)==self.extra_at and self.extra=='pixel':a[3,3,0]^=1
        return a
    def step(self,c):
        result=super().step(c)
        if len(self.commands)==self.extra_at:
            if self.extra=='body':self.agent.velocity=(1e-10,0.)
            elif self.extra=='goal':
                self.goal_state[0]+=1
                return result[0],result[1],result[2],result[3],self.info()
        return result


def execute(factory=ArrayEnv,*,p=None,data=None,store=None):
    fp,fd=fixture();p=fp if p is None else p;data=fd if data is None else data;s=MemoryStore() if store is None else store;envs=[]
    def make():e=factory();envs.append(e);return e
    def fake_sha(path):
        assert Path(path).name=='journal.jsonl'
        return m.source.digest(s.events)
    forbidden=AssertionError('sealed branch/execute/witness must NEVER be used')
    with patch.object(m.moving,'validate_env',lambda _:None),patch.object(m.source,'sample_state',lambda *a:oldtests.REQUEST.copy()),\
         patch.object(m,'sha',fake_sha),patch.object(m.base,'branch',side_effect=forbidden),\
         patch.object(m.base,'execute',side_effect=forbidden),patch.object(m.base.low,'branch',side_effect=forbidden),\
         patch.object(m.moving,'witness',side_effect=forbidden),patch.object(m.moving,'episode',side_effect=forbidden):
        r=m.execute(p,data,s,make,authorization=m.AUTH)
    return r,s,envs


def test_exact_prefix_then_new_suffix_and_full_census():
    r,s,envs=execute();p,d=fixture()
    assert r['status']=='moving_history_observed_suffix_physical_recorded',r
    assert not r['P1_admission'] and not r['P4_admission'] and not r['separate_old_control_reexecuted']
    assert r['new_neural_calls']==0 and r['new_independent_samples']==0
    expected=dict(reset=8,public_step=80,explicit_render=88,goal_render=8,set_state=16,physics_step=816,goal_bind=4)
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in expected.items()}
    for i,(row,e) in enumerate(zip(r['rows'],envs,strict=True)):
        assert row['prefix_exact'] and row['prefix_goal_exact'] and row['initial_exact'] and row['full_horizon']
        assert m.moving.exact(np.asarray(e.commands[:10]),m.moving.warmup_commands())
        assert m.moving.exact(np.asarray(e.commands[10:15]),m.base.commands(d['paired'][i]['original_commands'])[:5])
        assert m.moving.exact(np.asarray(e.commands[15:]),m.base.commands(p['plans'][i]['commands'])[5:])
        assert not np.array_equal(e.commands[15:],d['paired'][i]['original_commands'][5:])
        assert row['saved_old_terminal']==d['paired'][i]['old_terminal'] and e.closed
        assert row['paired_native_distance_delta']==row['final_metrics']['native_distance']-row['saved_old_final_metrics']['native_distance']
        assert [x['step'] for x in row['checkpoints']]==[0,5,10]
        checks=[x for x in s.events if x['kind']=='paired_prefix_exact' and x['branch']==i]
        assert [(x['phase'],x['step']) for x in checks]==[('warmup',n) for n in range(11)]+[('plan',n) for n in range(6)]
        admitted=next(x for x in s.events if x['kind']=='suffix_admitted_after_exact_prefix' and x['branch']==i)
        suffix=next(x for x in s.events if x['kind']=='operation_attempted' and x.get('branch')==i and x.get('phase')=='plan' and x.get('step')==6)
        assert admitted['index']<suffix['index']
    prev=None
    for e in s.events:
        assert e['previous_sha256']==prev and e['event_sha256']==m.source.digest({k:v for k,v in e.items() if k!='event_sha256'})
        prev=e['event_sha256']
    assert r['journal_chain_tail']==prev


@pytest.mark.parametrize('extra,at',[('pixel',12),('body',13),('goal',15)])
def test_real_prefix_drift_or_goal_drift_never_dispatches_suffix(extra,at):
    r,s,envs=execute(lambda:ArrayEnv(extra=extra,at=at))
    assert r['status']=='partial_or_failed' and len(r['rows'])==4 and len(envs)==4
    assert all(len(e.commands)==at for e in envs)
    assert all(not x['prefix_exact'] and x['last_observation']['png'] in s.files for x in r['rows'])
    assert not any(e['kind']=='suffix_admitted_after_exact_prefix' for e in s.events)


@pytest.mark.parametrize('field',['state','bits','pixel','png','goal'])
def test_saved_prefix_mismatch_is_failclosed_after_new_frame_persisted(field):
    p,d=fixture();ref=d['paired'][0]
    if field=='state':ref['prefix'][5]['state'][0]+=1e-8
    elif field=='bits':ref['prefix'][5]['snapshot']['state_float64_bits']='bad'
    elif field=='pixel':ref['prefix'][5]['pixel_sha256']='bad'
    elif field=='png':ref['prefix'][5]['file_sha256']='bad'
    else:ref['goal'][0]+=1
    r,s,envs=execute(p=p,data=d)
    assert len(envs[0].commands)==(10 if field=='goal' else 15)
    assert not r['rows'][0]['prefix_exact'] and r['rows'][0]['last_observation']['png'] in s.files
    assert r['rows'][1]['full_horizon'] and len(r['rows'])==4


def test_prefix_command_mutation_before_any_reset():
    p,d=fixture();p['plans'][0]['commands'][0][0]=.5
    r,_,envs=execute(p=p,data=d)
    assert envs[0].resets==[] and not envs[0].commands and r['rows'][1]['full_horizon']


@pytest.mark.parametrize('at',[5,12,15])
def test_raw_substitution_no_suffix_and_frame_kept(at):
    r,s,envs=execute(lambda:ArrayEnv(fault='substitute',at=at))
    assert all(len(e.commands)==at for e in envs) and all(not row['prefix_exact'] for row in r['rows'])
    assert all(row['last_observation']['png'] in s.files for row in r['rows'])


@pytest.mark.parametrize('fault',['terminal','truncate'])
def test_native_stop_in_suffix_no_padding_or_old_suffix_reexecution(fault):
    r,_,envs=execute(lambda:ArrayEnv(fault=fault,at=16))
    assert r['status']=='moving_history_observed_suffix_physical_recorded',r
    assert all(len(e.commands)==16 for e in envs) and all(x['prefix_exact'] and x['own_returned']==6 for x in r['rows'])
    assert r['counts']['physics_step']['completed']==656


def test_step_exception_after_prefix_preserves_primary_and_last_observed_prefix():
    r,s,envs=execute(lambda:ArrayEnv(fault='throw',at=16,close_error=True))
    assert all(e.closed for e in envs) and len(r['rows'])==4
    for row in r['rows']:
        assert row['prefix_exact'] and row['own_attempted']==6 and row['own_returned']==5
        assert 'primary step' in row['first_error']['message'] and 'secondary fake close' in str(row['secondary_errors'])
        assert row['last_observation']['step']==5 and row['last_observation']['png'] in s.files
    assert r['counts']['public_step']==dict(attempted=64,completed=60)


@pytest.mark.parametrize('fault',['guard','render'])
def test_guard_or_render_mutation_kept_before_rejection(fault):
    r,s,envs=execute(lambda:ArrayEnv(fault=fault,at=12))
    assert all(len(e.commands)==12 for e in envs) and r['status']=='partial_or_failed'
    assert all(x['last_observation']['png'] in s.files for x in r['rows'])


@pytest.mark.parametrize('fault',['setter','step','render','pose'])
def test_goal_attachment_forbids_extra_work_or_state_restore(fault):
    original_fixture()  # build independent OLD mock reference before injecting new-bind fault
    original=m.moving.attach_goal
    def attach(body,goal,rgb):
        original(body,goal,rgb)
        if fault=='setter':body._set_state(body._get_obs())
        elif fault=='step':body.space.step(.01)
        elif fault=='render':body.render()
        else:body.goal_pose[0]+=1
    with patch.object(m.moving,'attach_goal',attach):r,_,envs=execute()
    assert r['status']=='partial_or_failed' and all(len(e.commands)==10 for e in envs)
    assert all(x['own_attempted']==0 for x in r['rows'])


def test_caps_wall_storage_auth_no_dispatch():
    p,d=fixture();s=MemoryStore();ops=m.Ops(p,s);p['contract']['caps']['public_step']=0
    with pytest.raises(ValueError):ops.call('public_step',lambda:pytest.fail('dispatch'))
    ops.started-=1000
    with pytest.raises(ValueError):ops.call('reset',lambda:pytest.fail('dispatch'))
    with pytest.raises(ValueError):m.execute(p,d,s,lambda:pytest.fail('factory'),authorization='wrong')
    with pytest.raises(ValueError):m.run('x','x','x',lambda:pytest.fail('factory'),authorization='wrong')


def test_failed_goal_bind_charged_but_lexical_hook_scope_does_not_poison_next_branch():
    original_fixture();original=m.moving.attach_goal;calls=[]
    def attach(body,goal,rgb):
        calls.append(1)
        if len(calls)==1:raise RuntimeError('first branch attachment failed')
        return original(body,goal,rgb)
    with patch.object(m.moving,'attach_goal',attach):r,_,envs=execute()
    assert len(envs[0].commands)==10 and all(len(e.commands)==20 for e in envs[1:])
    assert r['counts']['goal_bind']==dict(attempted=4,completed=3)
    assert r['rows'][0]['status']=='failed' and all(x['prefix_exact'] for x in r['rows'][1:])
    assert r['status']=='partial_or_failed'


@pytest.mark.parametrize('stage',['branch_closed','run_finished','report.json'])
def test_durability_failure_cannot_report_success(stage):
    s=MemoryStore();put,emit=s.put,s.emit
    def put2(n,v,final=False):
        if n==stage:raise OSError('mock report write failure')
        return put(n,v,final)
    def emit2(kind,**kw):
        if kind==stage:raise OSError('mock journal failure')
        return emit(kind,**kw)
    s.put=put2;s.emit=emit2;r,_,envs=execute(store=s)
    assert r['status']=='partial_or_failed'
    if stage=='branch_closed':assert len(envs)==1 and all(x['status']=='not_started' for x in r['rows'][1:])


def test_final_binding_tamper_cannot_report_success():
    p,d=fixture();p['bindings']={'/MOCK/source.py':'old'}
    r,_,_=execute(p=p,data=d)
    assert r['status']=='partial_or_failed' and r['errors'][0]['stage']=='final_bindings'


def selection_fixture():
    p,d=fixture();consume=dict(schema=m.COMPILER+'.consume',status=m.CONSUME_STATUS,plans=[],errors=[],new_physics=0,P1_admission=False,P4_admission=False)
    handoff=consume;rows=[]
    mappings=[]
    for i,x in enumerate(p['plans']):
        s=x['root_slot'];v=dict(source_slot=s,arm=x['arm'],commands=x['commands'],blind_sha256=d['references'][s]['blind_sha256'],whole_run_report_sha256='f'*64)
        consume['plans'].append(deepcopy(v))
        mapping=dict(commands_pointer=f'/plans/{i}/commands',source_slot_pointer=f'/plans/{i}/source_slot',arm_pointer=f'/plans/{i}/arm',blind_sha_pointer=f'/plans/{i}/blind_sha256')
        mappings.append(mapping)
        row=dict(root_slot=s,arm=x['arm'],commands=x['commands'],compiler_report='/MOCK/own_consume.json',compiler_report_SHA='f'*64,
            compiler_schema=handoff['schema'],compiler_status=handoff['status'],
            **mapping)
        row['selection_digest']=m.source.digest(row);rows.append(row)
    authority=dict(source='/MOCK/'+m.COMPILER+'.py',source_sha256='f'*64,report='/MOCK/new_report.json',report_sha256='f'*64,
        consume='/MOCK/own_consume.json',consume_sha256='f'*64,consume_schema=consume['schema'],consume_status=consume['status'],mappings=mappings)
    manifest=dict(schema=m.NAME+'.selection',sourcepack='/MOCK/sourcepack.json',sourcepack_sha256='e'*64,
        source_prepared='/MOCK/source_prepare.json',source_prepare_sha256='d'*64,
        roots=[dict(root_slot=i,root_sha256=r['root_sha256']) for i,r in enumerate(d['references'])],arms=m.ARMS.copy(),plans=rows,
        selection_fixed_before_physics=True,physical_outcome_selection=False,wall_seconds=60,
        original_physical_report='/MOCK/original/report.json',original_physical_report_sha256=m.ORIGINAL_SHA,compiler_authority=authority)
    return manifest,d,handoff,consume


def checked_selection(a,d,h,c,sha='f'*64):
    def read(path):return c if Path(path).name=='own_consume.json' else h
    with patch.object(m,'sha',return_value=sha),patch.object(m,'read',side_effect=read),\
         patch.object(m.base.low,'sha',return_value=sha),patch.object(m.base.low,'read',side_effect=read):return m.selection(a,d)


def test_own_consume_layout_is_explicit_no_main_fabricated_wrapper():
    a,d,h,c=selection_fixture();rows,bindings=checked_selection(a,d,h,c)
    assert len(rows)==4 and len(bindings)==3 and rows[-1]['arm']==m.ARMS[1]


@pytest.mark.parametrize('field',['status','errors','blind_sha256'])
def test_missing_publisher_metadata_not_fabricated_by_main(field):
    a,d,h,c=selection_fixture()
    if field=='blind_sha256':del c['plans'][0][field]
    else:del c[field]
    with pytest.raises((KeyError,ValueError)):checked_selection(a,d,h,c)


def test_main_wrapper_cannot_replace_raw_own_consume_authority():
    a,d,h,c=selection_fixture();row=a['plans'][0];row['compiler_report']='/MOCK/wrapper.json'
    row['selection_digest']=m.source.digest({k:v for k,v in row.items() if k!='selection_digest'})
    with pytest.raises(ValueError):checked_selection(a,d,h,c)


@pytest.mark.parametrize('mutation',['prefix','consume_raw','consume_slot','consume_blind','consume_arm','running','errors','mapping','sha','missing','reorder','outcome'])
def test_selection_failclosed_mutations(mutation):
    a,d,h,c=selection_fixture()
    if mutation=='prefix':
        for v in (a['plans'][0]['commands'],c['plans'][0]['commands']):v[0][0]=.5
        a['plans'][0]['selection_digest']=m.source.digest({k:v for k,v in a['plans'][0].items() if k!='selection_digest'})
    elif mutation=='consume_raw':c['plans'][0]['commands'][6][0]=.75
    elif mutation=='consume_slot':c['plans'][0]['source_slot']=False
    elif mutation=='consume_blind':c['plans'][0]['blind_sha256']='bad'
    elif mutation=='consume_arm':c['plans'][0]['arm']='other'
    elif mutation=='running':c['status']='running';a['compiler_authority']['consume_status']='running'
    elif mutation=='errors':c['errors']=['mock failure']
    elif mutation=='mapping':a['compiler_authority']['mappings'][0]['commands_pointer']='/absent'
    elif mutation=='missing':a['plans'].pop()
    elif mutation=='reorder':a['plans'].reverse()
    elif mutation=='outcome':a['physical_outcome_selection']=True
    with pytest.raises(ValueError):checked_selection(a,d,h,c,sha='bad' if mutation=='sha' else 'f'*64)


@pytest.mark.parametrize('mutation',['warm','prefix','body','goal','census','full','order'])
def test_saved_original_adapter_rejects_incomplete_or_drifted_receipts(mutation):
    p,_,r,s,_=original_fixture();p=deepcopy(p);r=deepcopy(r);events=deepcopy(s.events)
    if mutation=='warm':next(x for x in events if x['kind']=='warmup_return')['raw_float32_bits']='bad'
    elif mutation=='prefix':next(x for x in events if x['kind']=='returned_transition' and x.get('phase')=='plan')['command']=[1,1]
    elif mutation=='body':next(x for x in events if x['kind']=='fresh_observation')['receipt']['snapshot']['state']=[]
    elif mutation=='goal':next(x for x in events if x['kind']=='goal_attached')['goal_rgb_sha256']='bad'
    elif mutation=='census':r['counts']['public_step']['completed']=79
    elif mutation=='full':r['rows'][0]['full_horizon']=False
    else:r['rows'].reverse()
    with pytest.raises(ValueError):m.assemble(r,p,events)


def test_saved_terminal_cannot_change_new_commands_or_prefix_gate():
    p,d=fixture();d['paired'][0]['old_final_metrics']['native_distance']=999.
    d['paired'][0]['old_terminal']['state']=[1.]*7
    r,_,envs=execute(p=p,data=d)
    assert r['status']=='moving_history_observed_suffix_physical_recorded'
    assert m.moving.exact(np.asarray(envs[0].commands[10:]),m.base.commands(p['plans'][0]['commands']))


def test_identical_selected_suffix_is_reported_not_forced_to_change():
    p,d=fixture();p['plans'][0]['commands']=deepcopy(d['paired'][0]['original_commands'])
    r,_,_=execute(p=p,data=d)
    assert r['status']=='moving_history_observed_suffix_physical_recorded' and r['rows'][0]['selected_suffix_matches_original']
    assert not r['separate_old_control_reexecuted']


def test_z_no_native_modules_imported():
    assert not any(n in sys.modules for n in ('gymnasium','pymunk','torch','stable_worldmodel'))
