"""Pure RAM signed arithmetic and accepted-receipt seams; no native/model calls."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_mixedmode_zero_signed_comparison_20261028_v1 as m
from test_agent_mixedmode_capability_physical_20261028_v1 import fixture as parent_fixture


def fixture():
    _,p,_=parent_fixture()
    for task in p['tasks']:
        root=np.asarray(task['native']['root_state']);goal=root+np.asarray([1,0,1,0,.5,0,0])
        task['native']['goal']=goal.tolist()
    jobs=m.zero.jobs_for(p);zrows=[];srows=[]
    for arm,dest in (('zero5',zrows),('selected_prefix5',srows)):
        originals=jobs if arm=='zero5' else p['jobs']
        for j in originals:
            task=p['tasks'][j['task']];native=task['native'];root=np.asarray(native['root_state'])
            selected=arm=='selected_prefix5';start=root.copy();start[0]+=.2 if selected else 0
            delta=np.zeros(7);command=np.asarray(j['probe_commands'][0])
            delta[:2]=command*(2 if selected else 1);delta[2:4]=command*(3 if selected else 1)
            delta[4]=command[0]*.5
            end=start+np.asarray([0,.3 if selected else .1,0,0,0,0,0])+delta
            snapshot=dict(state=start.tolist(),bodies={'block':{'velocity':[.25 if selected else 0,0]}})
            inv=dict(snapshot=snapshot,renderer=native['renderer'],goal_pose=native['goal_pose'])
            row=dict(deepcopy(j),status='completed',attempted=5,completed=5,error=None,
                condition=task['plan']['condition'],root_state=root.tolist(),goal=native['goal'],
                root_sha256=native['root_sha256'],goal_pose=native['goal_pose'],
                goal_pixel_sha256=native['goal_pixels'][task['plan']['condition']],
                start_verified=True,start_state=start.tolist(),end_state=end.tolist(),
                rgb_guard_accepted=True,frames=[dict(state=start.tolist(),pre=deepcopy(inv),post=deepcopy(inv)) for _ in range(6)])
            if selected:row.update(causal_prefix_commands=deepcopy(j['prefix_commands']),causal_replay_verified=True)
            else:row['zero5_executed']=True
            dest.append(row)
    counts=lambda caps:{('internal_physics_step' if k=='physics_step' else k):dict(attempted=v,completed=v) for k,v in caps.items()}
    zr=dict(schema=m.zero.NAME+'.report',status='complete',errors=[],fixed_denominator=20,rows=zrows,
        contract=m.zero.contract(),counts=counts(m.zero.CAPS),P1_admission=False,
        execution_binding=dict(prepared='/RAM/zero.json',prepare_sha256='1'*64,probe_prepared='/RAM/probe.json',probe_prepare_sha256=m.PROBE_SHA),
        executor_source_sha256=m.ZERO_SOURCE_SHA,journal={'file':'journal.jsonl','sha256':'a'*64},journal_tail='c'*64)
    sr=dict(schema=m.probe.NAME+'.evidence',executor_schema=m.physical.NAME+'.report',status='complete',errors=[],rows=srows,
        contract=m.physical.contract(),counts=counts(m.physical.CAPS),P1_admission=False,
        execution_binding=dict(prepared='/RAM/selected.json',probe_prepared='/RAM/probe.json',probe_prepare_sha256=m.PROBE_SHA),
        prepare_sha256=m.PROBE_SHA,executor_prepare_sha256='2'*64,executor_source_sha256=m.math.PHYSICAL_SOURCE_SHA,
        journal={'file':'journal.jsonl','sha256':'b'*64},journal_tail='d'*64)
    zp=dict(schema=m.zero.NAME+'.prepare',contract=m.zero.contract(),jobs=jobs,probe_digest=m.probe.digest(p),
        probe_prepare_sha256=m.PROBE_SHA,probe_prepared='/RAM/probe.json')
    flags=dict(P1_admission=False,independent_confirmation=False,new_neural_calls=0,new_physics_calls=0)
    zw=dict(schema=m.zero.NAME+'.consume',status='zero_time_control_responses_verified_not_admitted',fixed_denominator=20,
        report_sha256=m.ZERO_REPORT_SHA,prepare_sha256='1'*64,rows=deepcopy(zrows),counts=deepcopy(zr['counts']),errors=[],**flags)
    sc=dict(schema=m.probe.NAME+'.consume',status='saved_probe_responses_not_admitted',fixed_denominator=40,
        executor_report_sha256=m.SELECTED_REPORT_SHA,evidence_sha256=m.SELECTED_REPORT_SHA,
        executor_prepare_sha256='2'*64,prepare_sha256=m.PROBE_SHA,journal_sha256='b'*64,
        executor_source_sha256=m.math.PHYSICAL_SOURCE_SHA,rows=deepcopy(srows),counts=deepcopy(sr['counts']),errors=[],**flags)
    sw=dict(schema='MAIN.mixedmode_capability_acceptance.v1',receipt=sc,**flags)
    return p,zp,zr,zw,sr,sw


def pairs():
    p,zp,zr,zw,sr,sw=fixture();return p,m.matched_rows(p,zp,zr,sr)


def events_for(pairings):
    out=[[],[]]
    for j,z,s in pairings:
        for ai,row in enumerate((z,s)):
            events=out[ai];prefix=j['intervention_commands'] if ai==0 else s['causal_prefix_commands']
            raw=np.concatenate([m.zero.moving.warmup_commands(),m.probe.raw(prefix,5),m.probe.raw(j['probe_commands'],5)])
            for i,a in enumerate(raw):
                phase='warmup' if i<10 else ('zero' if ai==0 else 'prefix') if i<15 else 'probe'
                state=row['root_state'] if i<10 else row['start_state'] if i<15 else row['end_state']
                events.append(dict(kind='raw_return',job_digest=row['job_digest'],phase=phase,step=i+1 if i<15 else i-14,
                    raw_bits=a.tobytes().hex(),actual_bits=a.tobytes().hex(),actual_dtype=np.dtype('<f4').str,
                    state=state,terminated=False,truncated=False))
            for k,v in m.zero.CAPS.items():
                for _ in range(v//20):
                    for kind in ('operation_attempted','operation_returned'):
                        events.append(dict(kind=kind,job_digest=row['job_digest'],resource='internal_physics_step' if k=='physics_step' else k))
    return out


def test_all_fixed_spaces_probes_directions_counts_and_matched_elapsed():
    p,pairings=pairs();r=m.compare(p,pairings)
    assert len(r['groups'])==8 and len(r['comparisons'])==4
    assert r['denominators']['signed_projections']==160 and r['denominators']['signed_secants']==64
    assert all(list(g['spaces'])==list(m.SPACES) for g in r['groups'])
    counts=m.check_elapsed(pairings,*events_for(pairings))
    assert counts['zero5']==counts['selected_prefix5']
    assert counts['zero5']['public_step']==dict(attempted=400,completed=400)
    assert counts['zero5']['internal_physics_step']['completed']==4080
    assert all(c['elapsed_matched'] and not c['state_mediation_isolated'] for c in r['comparisons'])
    assert not r['P1_admission'] and not r['equal_total_compute']


def test_hand_signed_secants_and_response_difference_not_unsigned():
    p,pairings=pairs();r=m.compare(p,pairings);a=float(np.float32(.1))
    z,s=r['groups'][:2]
    assert z['spaces']['object_xy']['signed_original_task_secants']==pytest.approx([1,0])
    assert s['spaces']['object_xy']['signed_original_task_secants']==pytest.approx([3,0])
    diffs=r['comparisons'][0]['spaces']['object_xy']
    assert diffs['signed_secant_changes']==pytest.approx([2,0])
    assert diffs['signed_response_changes'][1]['selected_minus_zero']==pytest.approx(2*a)
    assert diffs['signed_response_changes'][2]['selected_minus_zero']==pytest.approx(-2*a)
    for _,_,row in pairings:row['end_state'][2]=2*row['start_state'][2]-row['end_state'][2]
    negative=m.compare(p,pairings)
    assert negative['comparisons'][0]['spaces']['object_xy']['signed_secant_changes'][0]==pytest.approx(-4.)


def test_zero_probe_drift_separate_and_object_only_not_full_task():
    p,pairings=pairs()
    for _,z,s in pairings:z['end_state'][2:4]=z['start_state'][2:4];s['end_state'][2:4]=s['start_state'][2:4]
    r=m.compare(p,pairings)
    for g in r['groups']:
        assert g['spaces']['object_xy']['signed_original_task_secants']==[0.,0.]
        assert g['spaces']['joint_xy']['signed_original_task_secants'][0]>0
        assert not g['spaces']['object_xy']['full_task_success_inferred']
        assert g['spaces']['agent_xy']['finite_responses'][0]['signed_original_task_projection']==0
        assert g['spaces']['agent_xy']['zero_drift'][1]>0


@pytest.mark.parametrize('shift',[-4*np.pi,-2*np.pi,2*np.pi,4*np.pi])
def test_wrapped_angle_invariant_not_2pi_geometry_error(shift):
    p,pairings=pairs();r=m.compare(p,pairings)
    for task in p['tasks']:task['native']['goal'][4]+=shift
    out=m.compare(p,pairings)
    assert [g['spaces']['object_angle_wrapped']['signed_original_task_secants'] for g in out['groups']]==[
        g['spaces']['object_angle_wrapped']['signed_original_task_secants'] for g in r['groups']]


def test_zero_original_direction_null_no_after_residual_refit():
    p,pairings=pairs()
    for task in p['tasks']:task['native']['goal']=deepcopy(task['native']['root_state'])
    r=m.compare(p,pairings)
    assert all(g['spaces'][s]['original_task_direction'] is None for g in r['groups'] for s in m.SPACES)
    assert all(v is None for c in r['comparisons'] for s in m.SPACES for v in c['spaces'][s]['signed_secant_changes'])


@pytest.mark.parametrize('field',['task','condition','context','goal','root','goalpixel','ancestry','zero','amplitude','missing',
    'selectedrole','selectedprefix','partial','renderer','snapshot','prepare','reorder'])
def test_matching_failclosed_not_cohort_filter(field):
    p,zp,zr,_,sr,_=fixture();z=zr['rows'][0];s=sr['rows'][5]
    if field=='task':z['task']=1
    elif field=='condition':z['condition']='on'
    elif field=='context':z['context_digest']='bad'
    elif field=='goal':z['goal']=[99.]*7
    elif field=='root':s['root_state']=[99.]*7
    elif field=='goalpixel':z['goal_pixel_sha256']='bad'
    elif field=='ancestry':z['parent_after_job_digest']='bad'
    elif field=='zero':z['intervention_commands'][0][0]=float(np.float32(.1))
    elif field=='amplitude':z['probe_commands'][0][0]=float(np.float32(.2))
    elif field=='missing':zr['rows'].pop()
    elif field=='selectedrole':s['role']='before'
    elif field=='selectedprefix':s['causal_prefix_commands'][0][0]=float(np.float32(.2))
    elif field=='partial':z['status']='native_stopped'
    elif field=='renderer':z['frames'][0]['pre']['renderer']['action_scale']=1
    elif field=='snapshot':z['frames'][0]['state'][0]+=1
    elif field=='prepare':sr['execution_binding']['probe_prepare_sha256']='0'*64
    else:zr['rows'].reverse()
    with pytest.raises((ValueError,KeyError)):m.matched_rows(p,zp,zr,sr)


@pytest.mark.parametrize('field',['zero_sha','old_sha','envelope','nested','rows','counts','partial','P1','calls'])
def test_two_real_receipt_layouts_strict(field):
    _,_,zr,zw,sr,sw=fixture()
    m.accepted_receipts(zr,zw,sr,sw,m.ZERO_REPORT_SHA,m.SELECTED_REPORT_SHA)
    if field=='zero_sha':zw['report_sha256']='bad'
    elif field=='old_sha':sw['receipt']['executor_report_sha256']='bad'
    elif field=='envelope':sw['schema']=m.probe.NAME+'.consume'
    elif field=='nested':sw['receipt']['schema']='guess'
    elif field=='rows':zw['rows'][0]['end_state'][0]+=1
    elif field=='counts':zw['counts']['public_step']['completed']-=1
    elif field=='partial':sw['receipt']['status']='partial'
    elif field=='P1':zw['P1_admission']=True
    else:sw['receipt']['new_neural_calls']=1
    with pytest.raises((ValueError,KeyError)):m.accepted_receipts(zr,zw,sr,sw,m.ZERO_REPORT_SHA,m.SELECTED_REPORT_SHA)


@pytest.mark.parametrize('field',['dropzero','raw','dtype','terminal','phase','start','resource'])
def test_actual_elapsed_not_string_claim(field):
    p,pairings=pairs();events=events_for(pairings)
    raw=[e for e in events[0] if e['kind']=='raw_return']
    if field=='dropzero':events[0].remove(raw[10])
    elif field=='raw':raw[10]['actual_bits']='bad'
    elif field=='dtype':raw[10]['actual_dtype']='<f8'
    elif field=='terminal':raw[10]['terminated']=True
    elif field=='phase':raw[10]['phase']='prefix'
    elif field=='start':raw[14]['state']=[999.]*7
    else:events[0].remove(next(e for e in events[0] if e['kind']=='operation_returned'))
    with pytest.raises(ValueError):m.check_elapsed(pairings,*events)


def test_within_arm_start_must_match_but_between_arm_state_not_required():
    p,pairings=pairs();out=m.compare(p,pairings)
    assert all(not c['state_matched'] and not c['snapshot_matched'] for c in out['comparisons'])
    pairings[1][1]['start_state'][0]+=1
    with pytest.raises(ValueError):m.compare(p,pairings)


def test_production_assemble_RAM_end_to_end(monkeypatch):
    p,zp,zr,zw,sr,sw=fixture();pairings=m.matched_rows(p,zp,zr,sr);events=events_for(pairings)
    sp=dict(schema=m.physical.NAME+'.prepare',contract=m.physical.contract(),probe_digest=zp['probe_digest'],
        probe_prepared='/RAM/probe.json',probe_prepare_sha256=m.PROBE_SHA)
    objects={'/RAM/z/report.json':zr,'/RAM/zreceipt.json':zw,'/RAM/s/report.json':sr,'/RAM/sreceipt.json':sw,
        '/RAM/zero.json':zp,'/RAM/selected.json':sp,'/RAM/probe.json':p}
    hashes={'/RAM/z/report.json':m.ZERO_REPORT_SHA,'/RAM/zreceipt.json':m.ZERO_RECEIPT_SHA,
        '/RAM/s/report.json':m.SELECTED_REPORT_SHA,'/RAM/sreceipt.json':m.SELECTED_RECEIPT_SHA,
        '/RAM/zero.json':'1'*64,'/RAM/selected.json':'2'*64,'/RAM/probe.json':m.PROBE_SHA,
        '/RAM/z/journal.jsonl':'a'*64,'/RAM/s/journal.jsonl':'b'*64,
        str(Path(m.zero.__file__).resolve()):m.ZERO_SOURCE_SHA,str(Path(m.physical.__file__).resolve()):m.math.PHYSICAL_SOURCE_SHA}
    monkeypatch.setattr(m,'sha',lambda path:hashes.get(str(path),'f'*64))
    monkeypatch.setattr(m.probe,'read',lambda path:deepcopy(objects[str(path)]))
    monkeypatch.setattr(m.physical,'read_events',lambda path,tail:events[0 if '/z/' in str(path) else 1])
    monkeypatch.setattr(m.physical,'consume',lambda *_:pytest.fail('no duplicate old consume job'))
    monkeypatch.setattr(m.zero,'consume',lambda *_:pytest.fail('receipt already accepted'))
    r=m.assemble('/RAM/z','/RAM/zreceipt.json','/RAM/s','/RAM/sreceipt.json')
    assert r['accepted_strict_receipt_authority'] and not r['full_producer_consume_reexecuted']
    assert r['matched_branch_resources']['zero5']['public_step']['completed']==400
    assert json.loads(m.canonical(r))==r and len(m.canonical(r))<m.CAP


def test_wrong_external_pair_fails_before_input_reads(monkeypatch):
    monkeypatch.setattr(m,'sha',lambda _:pytest.fail('must reject before IO'))
    with pytest.raises(ValueError):m.assemble('x','x','x','x',zero_report_sha='0'*64)


def test_cli_help_only(capsys):
    with pytest.raises(SystemExit) as exc:m.main(['--help'])
    assert exc.value.code==0 and '--zero-receipt-sha' in capsys.readouterr().out
