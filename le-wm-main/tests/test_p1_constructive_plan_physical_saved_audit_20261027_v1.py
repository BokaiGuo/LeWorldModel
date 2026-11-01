"""Only explicitly authorized saved evidence and in-RAM mutations. No tmp files."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_plan_physical_saved_audit_20261027_v1 as a

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'results/p1_constructive_plan_physical_20261027_v1_main01/report.json'
REPORT_SHA='d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300'


@pytest.fixture(scope='module')
def saved():
    r=a.read(a.bound(REPORT,REPORT_SHA));b=a.read(REPORT.parent/'execution_binding.json');p=a.read(a.bound(b['prepared'],b['prepare_sha256']))
    manifest=a.read(a.bound(p['manifest'],p['manifest_sha256']));refs=a.references(manifest);events=a.chain((REPORT.parent/'journal.jsonl').read_bytes())
    images={}
    for e in events:
        if e['kind']=='fresh_observation':
            receipt=e['receipt']
            with a.Image.open(a.bound(REPORT.parent/receipt['png'],receipt['file_sha256'])) as im:images[receipt['png']]=np.asarray(im).copy()
    return r,p,events,refs,images


def inspect(saved):
    r,p,ev,refs,images=saved
    return a.inspect(r,p,ev,refs,lambda receipt:images[receipt['png']])


def test_actual_full_saved_bindings_and_independent_numbers():
    b=a.read(REPORT.parent/'execution_binding.json');result=a.audit(REPORT,REPORT_SHA,b['prepared'],b['prepare_sha256'])
    assert result['status']=='complete_saved_protocol_verified' and result['findings']==[]
    assert result['PNG']==88 and result['counts']['physics_step']==dict(attempted=816,completed=816)
    assert result['counts']['public_step']==dict(attempted=80,completed=80)
    assert [r['final_metrics']['native_success'] for r in result['rows']]==[False]*4
    assert [r['final_metrics']['object_position_error'] for r in result['rows']]==pytest.approx([47.976995121251164,64.25127865978467,130.92915162616924,45.39850942624763])
    assert all(set(r['checkpoint_metrics'])=={0,5,10} for r in result['rows'])
    assert not result['P1_admission'] and result['new_physics']==result['new_model_calls']==0


def test_hashchain_mutation_duplicate_json_partial_line(saved):
    events=saved[2];raw=b''.join((a.json.dumps(e)+'\n').encode() for e in events)
    assert len(a.chain(raw))==len(events)
    bad=deepcopy(events);bad[2]['kind']='tampered'
    with pytest.raises(ValueError):a.chain(b''.join((a.json.dumps(e)+'\n').encode() for e in bad))
    with pytest.raises(ValueError):a.chain(raw[:-1])
    with pytest.raises(ValueError):a.decode(b'{"x":1,"x":2}')


@pytest.mark.parametrize('mutation',['commands','native','reward','goal','png','guard','counts','root','recipe','caps','denominator','last_observation','checkpoint','closed','per_step'])
def test_RAM_evidence_mutations_are_not_accepted(saved,mutation):
    r,p,ev,refs,images=deepcopy(saved)
    transition=next(e for e in ev if e['kind']=='returned_transition' and e['phase']=='plan')
    observation=next(e for e in ev if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan')
    if mutation=='commands':transition['command'][0]+=0.01
    elif mutation=='native':transition['actual_native_success']=not transition['actual_native_success']
    elif mutation=='reward':transition['reward']+=1
    elif mutation=='goal':transition['returned_goal_state'][2]+=1
    elif mutation=='png':images[observation['receipt']['png']][3,3,0]^=1
    elif mutation=='guard':next(e for e in ev if e['kind']=='guard_receipt')['accepted']=False
    elif mutation=='counts':r['counts']['physics_step']['completed']-=1
    elif mutation=='root':r['rows'][0]['root_state'][0]+=1
    elif mutation=='recipe':next(e for e in ev if e['kind']=='placement_request')['requested'][0]+=1
    elif mutation=='caps':p['contract']['caps']['physics_step']+=1
    elif mutation=='denominator':r['rows'].pop()
    elif mutation=='last_observation':r['rows'][0]['last_observation']['step']=5
    elif mutation=='checkpoint':r['rows'][0]['checkpoints'].pop()
    elif mutation=='closed':next(e for e in ev if e['kind']=='branch_closed')['receipt']['status']='failed'
    elif mutation=='per_step':
        first=next(i for i,e in enumerate(ev) if e['kind']=='operation_attempted' and e['resource']=='public_step')
        pair=ev[first+1:first+3];del ev[first+1:first+3]
        second=next(i for i,e in enumerate(ev) if e['kind']=='operation_attempted' and e['resource']=='public_step' and e['step']==2)
        ev[second+1:second+1]=pair;rechain(ev)  # same totals, 9 then11 substeps: must reject.
    result=inspect((r,p,ev,refs,images))
    assert result['status']=='saved_audit_failed' and result['findings']


def rechain(events):
    previous=None
    for index,e in enumerate(events):
        e.update(index=index,previous_sha256=previous);e['event_sha256']=a.digest({k:v for k,v in e.items() if k!='event_sha256'});previous=e['event_sha256']


@pytest.mark.parametrize('after_return',[False,True])
def test_step_or_render_internal_exception_has_NO_guaranteed_incident_frame(saved,after_return):
    r,p,ev,refs,images=deepcopy(saved)
    end=next(i for i,e in enumerate(ev) if e['kind']=='operation_attempted' and e['resource']=='public_step' and e.get('phase')=='plan')
    if after_return:
        end=next(i for i,e in enumerate(ev) if e['kind']=='operation_attempted' and e['resource']=='explicit_render' and any(x['kind']=='returned_transition' and x.get('phase')=='plan' for x in ev[:i]))
    ev=ev[:end+1]  # step or subsequent render enters then throws; no new frame.
    old=next(e['receipt'] for e in reversed(ev) if e['kind']=='fresh_observation')
    row=r['rows'][0]
    row.update(status='failed',first_error=dict(stage='branch',type='RuntimeError',message='RAM step internally threw'),secondary_errors=[],
        own_attempted=1,own_returned=int(after_return),warmup_returned=10,full_horizon=False,native_success=False,terminated=False,truncated=False,
        PNG=12,checkpoints=[old],last_observation=old)
    if after_return:
        transition=next(e for e in reversed(ev) if e['kind']=='returned_transition')
        row.update(final_state=transition['state'],final_metrics=transition['metrics'])
    for i in range(1,len(r['rows'])):
        plan=p['plans'][i]
        r['rows'][i]=dict(root_slot=plan['root_slot'],arm=plan['arm'],selection_digest=plan['selection_digest'],status='not_started',
            first_error=None,secondary_errors=[],own_attempted=0,own_returned=0,warmup_returned=0,initial_exact=False,PNG=0)
    counts={k:dict(attempted=0,completed=0) for k in a.KINDS}
    for e in ev:
        if e['kind'] in ('operation_attempted','operation_completed'):counts[e['resource']]['attempted' if e['kind']=='operation_attempted' else 'completed']+=1
    r.update(status='partial_or_failed',counts=counts,errors=[],native_successes=0)
    ev.append(dict(kind='branch_closed',receipt=deepcopy(row)))
    ev.append(dict(kind='run_finished',status=r['status'],counts=deepcopy(counts),errors=[]));rechain(ev)
    result=inspect((r,p,ev,refs,images))
    assert result['findings']==[],result['findings']
    assert result['status']=='partial_saved_no_success_claim'
    branch=result['rows'][0]
    assert branch['post_step_scene']==('unknown_after_unobserved_transition' if after_return else 'unknown_after_incomplete_step')
    assert branch['last_observation_role']=='pre_failure_last_observation_only'
    assert branch['incomplete_steps']==([] if after_return else [dict(phase='plan',step=1)])
    assert old['phase']=='plan' and old['step']==0  # must never be reported as fault step1.
    assert (branch['final_metrics'] is None)==(not after_return)


def test_metric_is_independent_native_bug_compatible():
    goal=np.zeros(7);state=np.zeros(7);state[4]=4*np.pi+1
    m=a.metrics(goal,state)
    assert m['negative_native_angle'] and m['native_success'] and not m['wrapped_allxy_success']
    assert m['object_success'] is None


def test_no_producer_core_or_neural_import():
    assert not any(n in sys.modules for n in ('torch','gymnasium','pymunk','stable_worldmodel'))
    import ast
    tree=ast.parse(Path(a.__file__).read_text())
    assert not any(isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('experiments') for n in ast.walk(tree))
