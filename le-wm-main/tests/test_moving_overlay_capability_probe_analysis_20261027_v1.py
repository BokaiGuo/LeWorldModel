"""Artificial RAM arithmetic/metadata only, never physical or scientific evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import moving_overlay_capability_probe_analysis_20261027_v1 as m


def h(text): return hashlib.sha256(str(text).encode()).hexdigest()


def snap(state, angle=None):
    state = np.asarray(state, dtype='<f8'); bodies = {}
    for name in ('agent', 'block'):
        x = np.zeros(9, dtype='<f8')
        x[:2] = state[:2] if name == 'agent' else state[2:4]
        if name == 'agent': x[2:4] = state[5:7]
        else: x[4] = state[4] if angle is None else angle
        bodies[name] = dict(position=x[:2].tolist(), velocity=x[2:4].tolist(), unwrapped_angle=float(x[4]),
            angular_velocity=float(x[5]), force=x[6:8].tolist(), torque=float(x[8]),
            is_sleeping=False, float64_bits=x.tobytes().hex())
    return dict(state=state.tolist(), state_float64_bits=state.tobytes().hex(), bodies=bodies)


def trace(state, step, angle=None, terminal=False):
    return dict(step=step,state=np.asarray(state,dtype=float).tolist(),snapshot=snap(state,angle),native_success=terminal,
                reward=None if step == 0 else float(terminal),terminated=terminal,truncated=False)


def events(counts):
    output = []
    for resource, count in counts.items():
        for i in range(count['attempted']):
            output.append(dict(resource=resource,kind='attempt'))
            output.append(dict(resource=resource,kind='return' if i<count['completed'] else 'failed'))
    return output


def fixture():
    w,prefix,steps = 10,5,2
    caps=dict(constructor=1,close=1,reset=2,public_step=17,physics_step=174,set_state=4,
              goal_setter=3,goal_render=2,goal_bind=1,explicit_render=54)
    roots=[dict(root_slot=s,task_id=h(('SYNTHETIC task',s)),source_root_state=[100.,100.,200.,200.,.2,0.,0.],
                goal_state=[120.,100.,200.,220.,.5,0.,0.]) for s in (0,1)]
    probes=[]
    for probe in m.PROBES:
        command=np.zeros((steps,2), dtype=np.float32)
        if probe!='zero': command[:,0 if probe[1]=='x' else 1] = (.25 if probe[1]=='x' else .5)*(1 if probe[0]=='+' else -1)
        probes.append(dict(probe_id=probe, commands=command.tolist()))
    cases=[]; rows=[]
    hashes=dict(prepare_sha256=h('SYNTHETIC prepare'),source_report_sha256=h('SYNTHETIC source'),planner_report_sha256=h('SYNTHETIC planner'))
    for s,c,a in m.case_keys():
        state=roots[s]['source_root_state'].copy(); state[0]+=m.ARMS.index(a)+2*m.CONDITIONS.index(c)
        start=dict(state=state,snapshot=snap(state),history_sha256=h((s,c,a,'history')))
        case=dict(root_slot=s,condition=c,arm=a,task_id=roots[s]['task_id'],selection_digest=h((s,c,a)),prefix_commands=np.full((prefix,2),.25,np.float32).tolist(),expected_start=start)
        cases.append(case)
        # Deterministic, intentionally artificial linear law. No simulator called.
        scale=1+m.ARMS.index(a)+2*m.CONDITIONS.index(c)
        for probe in probes:
            u=np.asarray(probe['commands'])[0]
            end=np.asarray(state).copy(); end[:2]+=np.array([-u[0],2*u[1]])*scale
            end[2:4]+=np.array([3*u[0],-4*u[1]])*scale; end[4]+=.1*u[0]-.2*u[1]
            trajectory=[trace(state,0),trace((np.asarray(state)+end)/2,1),trace(end,2)]
            counts={k:dict(attempted=v,completed=v) for k,v in caps.items()}
            rows.append(dict(root_slot=s,condition=c,arm=a,probe_id=probe['probe_id'],selection_digest=case['selection_digest'],
                task_id=roots[s]['task_id'],probe_sha256=m.digest(probe),**hashes,status='completed',start=deepcopy(start),
                prefix_commands=deepcopy(case['prefix_commands']),warmup_returned=w,trajectory=trajectory,
                returned_commands=deepcopy(probe['commands']),counts=counts,journal=events(counts)))
    protocol=dict(roots=roots,cases=cases,probes=probes,prefix_steps=prefix,probe_steps=steps,warmup_steps=w,meter_per_full_branch=caps)
    return dict(schema=m.PRODUCER+'.saved_analysis',units=deepcopy(m.UNITS),**hashes,protocol=protocol,rows=rows,
        source=dict(path='/RAM/physical/report.json',sha256=h('SYNTHETIC report')),prepare=dict(path='/RAM/physical/prepare.json',sha256=hashes['prepare_sha256']))


def missing(row, status='not_started'):
    row.update(status=status,start=None,prefix_commands=[],warmup_returned=0,trajectory=[],returned_commands=[],journal=[],
               counts={k:dict(attempted=0,completed=0) for k in m.RESOURCES})


def test_complete_linear_secants_units_signed_response_and_full_pairs():
    f=fixture(); original=deepcopy(f); result=m.analyze(f)
    assert f==original and result['status']=='analyzed'
    assert result['denominators']==dict(roots=2,cases=8,probes_per_case=5,branches=40,comparisons=8)
    assert len(result['rows'])==len(result['comparisons'])==8 and result['P1_admission'] is False
    r=result['rows'][0]['spaces']; obj=r['object_xy']; agent=r['agent_xy']
    np.testing.assert_array_equal(obj['matrix'],[[3.,0.],[0.,-4.]])
    np.testing.assert_array_equal(agent['matrix'],[[-1.,0.],[0.,2.]])
    np.testing.assert_allclose(r['object_angle_unwrapped']['matrix'],[[.1,-.2]],atol=1e-15)
    assert obj['spectrum']['singular_values']==[4.,3.] and obj['spectrum']['condition_number']==4/3
    assert obj['units']==agent['units']=='px/raw_command' and r['object_angle_unwrapped']['units']=='rad/raw_command'
    assert [x['task_projection'] for x in obj['finite_responses']]==[0.,0.,0.,-2.,2.]
    assert [x['task_projection'] for x in agent['finite_responses']]==[0.,-.25,.25,0.,0.]
    assert obj['projected_columns']==[0.,-4.] and obj['response_norm']==4.
    assert all(x==[0.,0.] for x in obj['asymmetry_columns'])
    np.testing.assert_array_equal(result['comparisons'][0]['spaces']['object_xy']['matrix_difference'],[[3.,0.],[0.,-4.]])
    np.testing.assert_array_equal(result['comparisons'][4]['spaces']['object_xy']['matrix_difference'],[[6.,0.],[0.,-8.]])
    assert all(not n['native_success'] for r in result['rows'] for n in r['native'])


def test_missing_signed_pair_never_drops_column_or_case():
    f=fixture(); missing(f['rows'][1]); result=m.analyze(f)
    assert result['status']=='analyzed_with_missing' and len(result['rows'])==8 and len(result['comparisons'])==8
    obj=result['rows'][0]['spaces']['object_xy']
    assert obj['columns'][0] is None and obj['columns'][1]==[0.,-4.] and obj['matrix'] is None and obj['spectrum'] is None
    assert obj['finite_responses'][1]['delta'] is None and obj['finite_responses'][1]['task_projection'] is None
    assert obj['finite_responses'][3]['task_projection']==-2.
    assert result['comparisons'][0]['spaces']['object_xy']['matrix_difference'] is None


def test_missing_zero_keeps_signed_secants_but_not_zero_relative_response():
    f=fixture(); missing(f['rows'][0]); obj=m.analyze(f)['rows'][0]['spaces']['object_xy']
    assert obj['matrix']==[[3.,0.],[0.,-4.]] and obj['zero_drift'] is None
    assert obj['asymmetry_columns']==[None,None]
    assert all(x['delta'] is None and x['task_projection'] is None for x in obj['finite_responses'])


def test_native_early_stop_retains_native_flags_and_missing_fixed_endpoint():
    f=fixture(); row=f['rows'][1]; row['status']='native_stopped'; row['trajectory']=row['trajectory'][:2]
    row['trajectory'][-1].update(terminated=True,native_success=True,reward=1.)
    row['returned_commands']=row['returned_commands'][:1]
    row['counts']['public_step']=dict(attempted=16,completed=16)
    row['counts']['physics_step']=dict(attempted=164,completed=164)
    row['counts']['explicit_render']=dict(attempted=51,completed=51); row['journal']=events(row['counts'])
    r=m.analyze(f)['rows'][0]
    assert r['missing_probes']==['+x'] and r['native'][1]['native_success'] is True
    assert r['native'][1]['returned_steps']==1 and not r['native'][1]['fixed_endpoint_available']


def test_zero_response_and_zero_task_direction_are_distinct():
    f=fixture()
    for r in f['rows']:
        r['trajectory']=[trace(r['start']['state'],step) for step in range(3)]
    result=m.analyze(f); obj=result['rows'][0]['spaces']['object_xy']
    assert obj['matrix']==[[0.,0.],[0.,0.]] and obj['spectrum']['singular_values']==[0.,0.]
    assert obj['spectrum']['condition_number'] is None and obj['spectrum']['condition_status']=='singular'
    assert obj['response_norm']==0 and all(x['task_projection']==0 for x in obj['finite_responses'])
    for root in f['protocol']['roots']: root['goal_state']=root['source_root_state'].copy()
    obj=m.analyze(f)['rows'][0]['spaces']['object_xy']
    assert obj['task_direction'] is None and obj['response_norm'] is None
    assert all(x['task_projection'] is None for x in obj['finite_responses'])


def test_real_preprobe_native_stop_keeps_missing_case_and_stop_evidence():
    f=fixture(); row=f['rows'][0]; missing(row,'native_stopped')
    row['warmup_returned']=3
    state=f['protocol']['roots'][0]['source_root_state']
    row['preprobe_stop']=dict(phase='warmup',step=3,state=state,snapshot=snap(state),
                             native_success=True,terminated=True,truncated=False)
    completed=dict(constructor=1,close=1,reset=2,public_step=3,physics_step=34,set_state=4,
                   goal_setter=2,goal_render=2,goal_bind=0,explicit_render=12)
    row['counts']={k:dict(attempted=v,completed=v) for k,v in completed.items()}; row['journal']=events(row['counts'])
    result=m.analyze(f)
    assert result['rows'][0]['missing_probes']==['zero']
    assert result['rows'][0]['native'][0]['preprobe_stop']==row['preprobe_stop']
    assert result['rows'][0]['native'][0]['native_success'] is None
    assert all(x['task_projection'] is None for x in result['rows'][0]['spaces']['object_xy']['finite_responses'])
    row['preprobe_stop']['step']=2
    with pytest.raises(ValueError): m.analyze(f)


def test_failed_dispatch_preserves_attempt_vs_return_without_imputed_state():
    f=fixture(); row=f['rows'][0]; missing(row,'engineering_failed')
    row['counts']['constructor']['attempted']=1; row['journal']=events(row['counts'])
    result=m.analyze(f)
    assert result['rows'][0]['native'][0]['native_success'] is None
    assert result['rows'][0]['spaces']['object_xy']['zero_drift'] is None


def test_partial_meter_lie_rejected_even_when_ledger_matches():
    f=fixture(); row=f['rows'][0]; missing(row,'engineering_failed')
    row['counts']['physics_step']=dict(attempted=1,completed=1)
    row['journal']=events(row['counts'])
    with pytest.raises(ValueError): m.analyze(f)


def test_wrap_cut_uses_unwrapped_body_angle_not_modulo_state_difference():
    f=fixture()
    for row in f['rows'][:5]:
        for step,t in enumerate(row['trajectory']):
            a=.01 + (3.2 if row['probe_id']=='+x' else -3.2 if row['probe_id']=='-x' else 0.)*step/2
            t['state'][4]=a%(2*np.pi); t['snapshot']=snap(t['state'],a)
        if row['probe_id']=='zero': continue
    # step0 must match frozen start; change just the final angular endpoints instead.
    for row in f['rows'][:5]:
        row['trajectory'][0]=trace(row['start']['state'],0)
    r=m.analyze(f)['rows'][0]['spaces']
    assert r['object_angle_unwrapped']['angle_branch_cut_ambiguous'][0] is True
    assert r['object_angle_unwrapped']['matrix'][0][0]==pytest.approx(12.8)
    assert abs(r['object_angle_wrapped_diagnostic']['matrix'][0][0])<1
    assert r['object_angle_wrapped_diagnostic']['diagnostic_only'] is True


@pytest.mark.parametrize('change',[
    lambda f:f['rows'].pop(), lambda f:f['rows'].reverse(),
    lambda f:f['rows'][0].update(root_slot=True),
    lambda f:f['rows'][0].update(task_id=h('other')),
    lambda f:f['rows'][0].update(selection_digest=h('other')),
    lambda f:f['rows'][0].update(probe_sha256=h('other')),
    lambda f:f['rows'][0].update(source_report_sha256=h('other')),
    lambda f:f['rows'][0].update(planner_report_sha256=h('other')),
    lambda f:f['rows'][0].update(prepare_sha256=h('other')),
    lambda f:f['rows'][0]['start'].update(history_sha256=h('other')),
    lambda f:f['rows'][0]['start']['state'].__setitem__(0,100.00000000000001),
    lambda f:f['rows'][0]['start']['snapshot']['bodies']['block'].update(torque=1.),
    lambda f:f['rows'][0]['trajectory'][-1].update(step=1),
    lambda f:f['rows'][0]['trajectory'][0].update(terminated=True),
    lambda f:f['rows'][0]['trajectory'][-1].update(native_success=0),
    lambda f:f['rows'][0]['trajectory'][-1].update(state=[0.]*5),
    lambda f:f['rows'][0]['counts']['public_step'].update(completed=16),
    lambda f:f['rows'][0]['counts'].pop('goal_bind'),
    lambda f:f['rows'][0]['journal'].pop(),
    lambda f:f['rows'][0]['journal'][0].update(kind='return'),
    lambda f:f['rows'][0].update(returned_commands=[[.1,0.],[.1,0.]]),
    lambda f:f['rows'][0]['prefix_commands'][0].__setitem__(0,.5),
    lambda f:f['rows'][0].update(status='not_started'),
    lambda f:f['protocol'].update(probe_steps=3),
    lambda f:f['protocol']['probes'].reverse(),
    lambda f:f['protocol']['probes'][0]['commands'][0].__setitem__(0,.25),
    lambda f:f['protocol']['probes'][1]['commands'][0].__setitem__(1,.25),
    lambda f:f['protocol']['meter_per_full_branch'].update(physics_step=170),
    lambda f:f['units'].update(position='m'),
])
def test_mutations_refused_without_filtering_denominator(change):
    f=fixture(); change(f)
    with pytest.raises((ValueError,KeyError)): m.analyze(f)


def test_saved_closure_hashes_protocol_rows_and_external_authority(monkeypatch):
    f=fixture(); f['source']['sha256']=h('report bytes'); f['prepare']['sha256']=f['prepare_sha256']
    report=dict(schema=m.PRODUCER+'.report',analysis_protocol=f['protocol'],analysis_rows=f['rows'],
        **{k:f[k] for k in ('prepare_sha256','source_report_sha256','planner_report_sha256')})
    prep=dict(schema=m.PRODUCER+'.prepare',analysis_protocol=f['protocol'],
        **{k:f[k] for k in ('source_report_sha256','planner_report_sha256')})
    manifest_path='/RAM/manifest.json'; objects={manifest_path:f,f['source']['path']:report,f['prepare']['path']:prep}
    hashes={manifest_path:h('manifest bytes'),f['source']['path']:f['source']['sha256'],f['prepare']['path']:f['prepare_sha256']}
    monkeypatch.setattr(m,'read',lambda p:deepcopy(objects[str(p)]))
    monkeypatch.setattr(m,'sha',lambda p:hashes[str(p)])
    assert len(m.load_saved(manifest_path,hashes[manifest_path])['input_bindings'])==3
    with pytest.raises(ValueError): m.load_saved(manifest_path,h('guessed future hash'))
    report['analysis_rows']=deepcopy(f['rows']); report['analysis_rows'][0]['task_id']=h('swapped')
    with pytest.raises(ValueError): m.load_saved(manifest_path,hashes[manifest_path])
    report['analysis_rows']=f['rows']; prep['analysis_protocol']=deepcopy(f['protocol']); prep['analysis_protocol']['probe_steps']=3
    with pytest.raises(ValueError): m.load_saved(manifest_path,hashes[manifest_path])


def test_read_duplicate_nan_and_size_rejected_without_disk(monkeypatch):
    class FakePath:
        def is_file(self): return True
        def stat(self): return type('Stat',(),dict(st_size=10))()
        def read_bytes(self): return payload[0]
    monkeypatch.setattr(m,'Path',lambda _:FakePath())
    for value in (b'{"a":1,"a":2}',b'{"a":NaN}'):
        payload=[value]
        with pytest.raises(ValueError): m.read('/RAM')


def test_no_neural_physical_import_and_no_fitting():
    assert not any(name in sys.modules for name in ('torch','gymnasium','pymunk','stable_worldmodel'))
    import ast
    tree=ast.parse(Path(m.__file__).read_text())
    imported=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(x and 'physical' in x for x in imported)
    assert not [n for n in ast.walk(tree) if isinstance(n,ast.Attribute) and n.attr in ('lstsq','pinv','fit','solve','execute','run')]


def native_fixture():
    """Build Nash-shaped RAM metadata without importing/running its producer."""
    f=fixture(); plans=[]; anchors=[]; old_rows=[]; native_rows=[]; ev=[]
    def emit(kind,**fields): ev.append(dict(kind=kind,**deepcopy(fields)))
    cap=dict(constructor=1,close=1,reset=2,public_step=20,physics_step=204,set_state=4,
             goal_setter=3,goal_render=2,goal_bind=1,explicit_render=66)
    pp=[]
    for name,d in zip(m.PROBES,((0,0),(.1,0),(-.1,0),(0,.1),(0,-.1)),strict=True):
        pp.append(dict(name=name,commands=np.tile(np.asarray(d,dtype=np.float32),(5,1)).tolist()))
    contract=dict(anchors=8,probes_per_anchor=5,fixed_denominator=40,independent_roots=2,
        warmup=10,prefix=5,suffix=5,raw_dtype='float32',action_scale=100,relative=True,native_position_clip=False,
        raw_action_bounds=[-1,1],state_dimension=7,no_neural=True,no_goal_pose_edit=True,P1_admission=False,
        probes=pp,caps={k:40*v for k,v in cap.items()})
    for ci,case in enumerate(f['protocol']['cases']):
        s,c,a=m.key(case); task=h(('native packet',s,c)); state=case['expected_start']['state']; snapshot=case['expected_start']['snapshot']
        plan=dict(root_slot=s,condition=c,arm=a,selection_digest=case['selection_digest'],task_id=task,
                  commands=case['prefix_commands']+np.zeros((5,2),np.float32).tolist(),
                  source_report_sha256=f['source_report_sha256'],planner_report_sha256=f['planner_report_sha256'])
        plans.append(plan)
        cp=dict(step=5,state=state,snapshot=snapshot)
        at=[]
        for phase,n in (('warmup',11),('plan',6)):
            for step in range(n):
                for overlay in range(3):
                    receipt=dict(phase=phase,step=step,condition='on' if overlay==1 else 'off',overlay_index=overlay,
                        file_sha256=h(('SYNTHETIC png',ci,phase,step,overlay)),pixel_sha256=h(('SYNTHETIC pixels',ci,phase,step,overlay)),
                        state=state,snapshot=snapshot)
                    invariant=dict(snapshot=snapshot)
                    at.append(dict(kind='fresh_observation',receipt=receipt,pre=invariant,post=invariant))
        anchors.append(dict(index=ci,checkpoint=cp,trace=at))
        old_rows.append(dict(**{k:plan[k] for k in ('root_slot','condition','arm','task_id','selection_digest')},
                             root_state=f['protocol']['roots'][s]['source_root_state'],goal=f['protocol']['roots'][s]['goal_state']))
        for bi,probe in enumerate(pp):
            where=dict(root_slot=s,condition=c,arm=a,probe=probe['name'])
            for e in events({k:dict(attempted=v,completed=v) for k,v in cap.items()}):
                emit({'attempt':'operation_attempted','return':'operation_completed'}[e['kind']],resource=e['resource'],**where)
            for e in at: emit(e['kind'],**where,**{k:v for k,v in e.items() if k!='kind'})
            emit('suffix_start_verified',**where)
            responses=[]
            warm=np.concatenate((np.tile(np.asarray([.1,0],dtype='<f4'),(5,1)),np.tile(np.asarray([0,.1],dtype='<f4'),(5,1))))
            for phase,cmds in (('warmup',warm),('plan',np.asarray(case['prefix_commands'],dtype='<f4')),('probe',np.asarray(probe['commands'],dtype='<f4'))):
                for step,command in enumerate(cmds,1):
                    current=np.asarray(state).copy()
                    if phase=='probe': current[2:4]+=command.astype(float)*step
                    emit('returned_transition',**where,phase=phase,step=step,requested_bits=command.tobytes().hex(),
                         actual_bits=command.tobytes().hex(),actual_dtype='<f4',state=current.tolist(),snapshot=snap(current),
                         reward=0.,terminated=False,truncated=False)
                    if phase=='probe':
                        delta=current-np.asarray(state)
                        responses.append(dict(step=step,state=current.tolist(),delta_native7=delta.tolist(),
                            delta_angle_wrapped=float(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))),terminated=False,truncated=False))
            native_rows.append(dict(index=ci*5+bi,anchor_index=ci,**where,selection_digest=case['selection_digest'],task_id=task,
                start_verified=True,start_state=state,warmup_returned=10,own_returned=10,status='probe_completed',responses=responses))
    p=dict(schema=m.PRODUCER+'.prepare',contract=contract,plans=plans,anchors=anchors,physical_output='/RAM/prior',
           physical_prepare_sha256=h('native prior prepare'),physical_report_sha256=h('native prior report'),
           planner_report_sha256=f['planner_report_sha256'],bindings={})
    prior=dict(schema='p1_moving_overlay_recompile_physical_20261027_v1.report',rows=old_rows,
        source_report_sha256=f['source_report_sha256'],planner_report_sha256=f['planner_report_sha256'],prepare_sha256=p['physical_prepare_sha256'])
    report=dict(schema=m.PRODUCER+'.report',contract=contract,rows=native_rows,prepare_sha256=f['prepare_sha256'],
        physical_report_sha256=p['physical_report_sha256'],planner_report_sha256=f['planner_report_sha256'],
        fixed_denominator=40,independent_roots=2,inputs_final_verified=True,new_neural_calls=0,P1_admission=False,
        status='capability_probes_recorded',errors=[],counts={k:dict(attempted=40*v,completed=40*v) for k,v in cap.items()})
    return p,report,ev,prior,f['prepare_sha256']


def test_actual_Nash_layout_adapter_preserves_per_condition_packet_ids_and_missing_native_success():
    p,r,e,prior,seal=native_fixture()
    saved=m.normalize_native(p,r,e,prior,seal); result=m.analyze(saved)
    assert result['status']=='analyzed' and len(result['rows'])==8
    assert saved['protocol']['cases'][0]['task_id']!=saved['protocol']['cases'][2]['task_id']
    assert all(n['native_success'] is None and n['terminated'] is False for row in result['rows'] for n in row['native'])
    assert result['probe_amplitudes_raw']==[float(np.float32(.1))]*2
    np.testing.assert_allclose(result['rows'][0]['spaces']['object_xy']['matrix'],np.eye(2)*5,atol=0,rtol=0)


@pytest.mark.parametrize('change',[
    lambda p,r,e,old:r['rows'].pop(),
    lambda p,r,e,old:r['counts']['physics_step'].update(completed=1),
    lambda p,r,e,old:r['rows'][0].update(task_id=h('wrong')),
    lambda p,r,e,old:r['rows'][0].update(start_state=[0.]*7),
    lambda p,r,e,old:p['contract']['probes'][1]['commands'][0].__setitem__(0,.2),
    lambda p,r,e,old:next(x for x in e if x['kind']=='returned_transition').update(actual_bits='00'),
    lambda p,r,e,old:next(x for x in e if x['kind']=='fresh_observation')['receipt'].update(file_sha256=h('wrong image')),
    lambda p,r,e,old:r['rows'][1]['responses'][0].update(delta_native7=[0.]*7),
    lambda p,r,e,old:old['rows'][2].update(goal=[0.]*7),
])
def test_native_layout_mutation_refusal(change):
    p,r,e,prior,seal=native_fixture(); change(p,r,e,prior)
    with pytest.raises((ValueError,KeyError)): m.analyze(m.normalize_native(p,r,e,prior,seal))


def test_native_journal_chain_uses_producer_hash_not_compact_normalized_hash(monkeypatch):
    ev=[]; previous=None
    for i,kind in enumerate(('run_started','run_finished')):
        row=dict(index=i,kind=kind,previous_sha256=previous)
        row['event_sha256']=m.native_digest(row); previous=row['event_sha256']; ev.append(row)
    payload=[b''.join(json.dumps(x).encode()+b'\n' for x in ev)]
    class FakePath:
        def stat(self): return type('S',(),dict(st_size=len(payload[0])))()
        def read_bytes(self): return payload[0]
    monkeypatch.setattr(m,'Path',lambda _:FakePath())
    assert m.read_journal('/RAM',previous)==ev
    ev[0]['kind']='mutated'; payload[0]=b''.join(json.dumps(x).encode()+b'\n' for x in ev)
    with pytest.raises(ValueError): m.read_journal('/RAM',previous)


def test_direct_native_loader_readonly_closure_and_wrong_external_sha(monkeypatch):
    p,r,e,prior,seal=native_fixture()
    directory=Path('/RAM/probe'); report_sha=h('actual RAM native report')
    r.update(journal_sha256=h('actual RAM journal'),journal_chain_tail=h('RAM tail'))
    p['bindings']={'/RAM/old.png':h('RAM old PNG')}
    ppath='/RAM/prepared.json'
    binding=dict(prepare_sha256=seal,physical_report_sha256=p['physical_report_sha256'],planner_report_sha256=p['planner_report_sha256'])
    objects={str(directory/'report.json'):r,ppath:p,'/RAM/prior/report.json':prior,str(directory/'execution_binding.json'):binding}
    hashes={str(directory/'report.json'):report_sha,ppath:seal,'/RAM/prior/report.json':p['physical_report_sha256'],
            str(directory/'journal.jsonl'):r['journal_sha256'],str(directory/'execution_binding.json'):h('RAM binding'),**p['bindings']}
    monkeypatch.setattr(m,'read',lambda path:deepcopy(objects[str(path)]))
    monkeypatch.setattr(m,'sha',lambda path:hashes[str(path)])
    monkeypatch.setattr(m,'read_journal',lambda path,tail:deepcopy(e))
    result=m.load_native(directory,report_sha,ppath,seal)
    assert result['status']=='analyzed' and result['input_bindings']['/RAM/old.png']==h('RAM old PNG')
    with pytest.raises(ValueError): m.load_native(directory,h('wrong report'),ppath,seal)
    hashes['/RAM/old.png']=h('mutated dependency')
    with pytest.raises(ValueError): m.load_native(directory,report_sha,ppath,seal)
