"""RAM/JSON/numpy fixtures only. No tmp files, model or native physics."""
from copy import deepcopy
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import moving_budget_scheduler_comparison_20261027_v1 as m


METRIC_KEYS=('native_success','native_distance','native_angle_term','negative_native_angle','wrapped_allxy_success',
    'wrapped_object_angle_error','object_position_error','object_pose','object_success')


def old_metric(goal,state):return {k:v for k,v in m.metrics(goal,state).items() if k in METRIC_KEYS}


def fixture(early=False,equivalent=False):
    methods=m.METHODS;h='c'*64;alloc=dict(zip(methods,([10,10],[10,10] if equivalent else [15,5],[10,10] if equivalent else [5,15],[10,10])))
    scopes={g:{} for g in ('shared',)+methods}
    for group in scopes:
        for slot in (0,1):
            d={k:0 for k in m.scheduler.MAXIMA}
            if group=='shared':
                d.update(dict.fromkeys(m.scheduler.NEURAL,4));d.update(jacobian=1,svd=1,vjp_directions=192,grad_api=192,raw_backward_hook=192)
            else:d.update(dict.fromkeys(m.scheduler.NEURAL,6*alloc[group][slot]+4))
            scopes[group][str(slot)]={k:dict(attempted=v,completed=v) for k,v in d.items()}
    plans=[]
    for i,arm in enumerate(methods):
        for s in (0,1):
            raw=np.full((10,2),np.float32((1 if equivalent else i+1)/100),np.float32).astype(float).tolist()
            p=dict(schema=m.scheduler.NAME+'.model_plan',source_slot=s,root_sha256=f'{s+1:064x}',task_id=f'{s+3:064x}',blind_sha256=f'{s+5:064x}',
                arm=arm,commands=raw,candidate_index=0 if equivalent else i,model_cost=1.+s-(0 if equivalent else i/10),physical_execution=False,selection_uses_physics=False,**m.scheduler.FLAGS)
            p['selection_sha256']=m.digest(p);p['whole_run_report_sha256']=h;plans.append(p)
    receipt=dict(schema=m.scheduler.NAME+'.consume',status='four_policy_commands_verified_not_admitted',errors=[],plans=plans,**m.scheduler.FLAGS)
    prepared_plans=[]
    for s in (0,1):
        for arm in methods:
            plan=next(p for p in plans if p['source_slot']==s and p['arm']==arm)
            p=dict(root_slot=s,root_sha256=plan['root_sha256'],task_id=plan['task_id'],blind_sha256=plan['blind_sha256'],arm=arm,
                commands=deepcopy(plan['commands']),compiler_authority=m.scheduler.NAME,compiler_report='/RAM/scheduler/report.json',
                compiler_report_sha256=h,compiler_selection_sha256=plan['selection_sha256'],own_plan_sha256=m.digest(plan))
            p['selection_digest']=m.digest(p);prepared_plans.append(p)
    pp=dict(schema=m.PHYSICAL+'.prepare',plans=prepared_plans,contract=dict(caps={k:0 for k in m.RESOURCES}),
        sourcepack_sha256=m.scheduler.SOURCEPACK_SHA,source_prepare_id='b'*64,bindings={},manifest='/RAM/manifest.json',manifest_sha256='d'*64)
    physical=dict(schema=m.PHYSICAL+'.report',status='moving_budget_scheduler_physical_recorded',errors=[],rows=[],fixed_branch_denominator=8,
        new_neural_calls=0,training=False,prepared='/RAM/physical_prepare.json',prepare_sha256='e'*64,contract=deepcopy(pp['contract']),
        sourcepack_sha256=m.scheduler.SOURCEPACK_SHA,source_prepare_id='b'*64,counts={k:dict(attempted=0,completed=0) for k in m.RESOURCES},
        total_internal_physics_scope='hook_observed_native_work',wall_seconds=2.,**m.FLAGS)
    events=[]
    def emit(kind,**fields):
        e=dict(index=len(events),previous_sha256=None if not events else events[-1]['event_sha256'],kind=kind,**deepcopy(fields));e['event_sha256']=m.digest(e);events.append(e)
    emit('run_started',prepare=pp,execution_binding=dict(prepared=physical['prepared'],prepare_sha256=physical['prepare_sha256']))
    for b,plan in enumerate(prepared_plans):
        s=plan['root_slot'];goal=[180.,180.,280.,280.,0.,0.,0.];root=[100.+s,100.,200.,200.,0.,0.,0.];state=root.copy()
        n=5 if early and b==1 else 10;stopped=n<10
        row=dict(branch_index=b,root_slot=s,arm=plan['arm'],selection_digest=plan['selection_digest'],selected_plan=deepcopy(plan),
            initial_exact=True,first_error=None,secondary_errors=[],constructor_internal_work='sealed_native_pristine_no_physics',
            warmup_returned=10,own_attempted=n,own_returned=n,full_horizon=not stopped,endpoint_scope='partial' if stopped else 'full10',
            status='native_stopped' if stopped else 'horizon_completed',native_success=False,terminated=False,truncated=stopped,
            goal=goal,root_state=root,initial_metrics=old_metric(goal,root),PNG=0,checkpoints=[])
        where=dict(branch=b,root_slot=s,arm=plan['arm'],selection_digest=plan['selection_digest'])
        def op(resource,phase=None,step=None,raw=None):
            fields={}
            if phase is not None:fields.update(phase=phase,step=step,raw_action=raw)
            physical['counts'][resource]['attempted']+=1;emit('operation_attempted',resource=resource,**where,**fields)
            return fields
        def ret(resource,fields):
            physical['counts'][resource]['completed']+=1;emit('operation_completed',resource=resource,**where,**fields)
        def simple(resource):f=op(resource);ret(resource,f)
        def capture(phase,step):
            simple('explicit_render');row['PNG']+=1
            frame=dict(phase=phase,step=step,png=f'b{b}_{phase}_{step:02d}.png',file_sha256='a'*64,pixel_sha256='9'*64,
                state=state.copy(),snapshot=dict(state=state.copy()),observation_state=state.copy())
            emit('fresh_observation',receipt=frame,**where)
            emit('guard_receipt',phase=phase,step=step,accepted=True,render_preserved_dynamics=True,canonical_agent_pixels=1,**where)
            if phase=='plan' and step in (0,5,10):row['checkpoints'].append(frame)
        simple('environment_construct')
        emit('branch_started',plan=plan,raw_warmup=m.scheduler.moving.fixed_raw().astype(float).tolist(),
            raw_float32_bits=m.scheduler.moving.fixed_raw().tobytes().hex(),**where)
        for _ in range(2):
            f=op('reset')
            for resource in ('set_state','goal_render','set_state'):
                ff=op(resource)
                if resource=='set_state':simple('physics_step')
                ret(resource,ff)
            ret('reset',f)
        capture('warmup',0)
        for step,raw in enumerate(m.scheduler.moving.fixed_raw().astype(float).tolist(),1):
            f=op('public_step','warmup',step,raw)
            for _ in range(10):simple('physics_step')
            ret('public_step',f);emit('returned_transition',phase='warmup',step=step,state=root,reward=-m.metrics(goal,root)['native_distance'],terminated=False,truncated=False,**where)
            capture('warmup',step)
        simple('goal_bind');capture('plan',0)
        for step,raw in enumerate(plan['commands'][:n],1):
            f=op('public_step','plan',step,raw)
            for _ in range(10):simple('physics_step')
            ret('public_step',f)
            # Synthetic arrays are NOT a simulator: deterministic named fixture differences.
            state=[root[0]+step*(1 if equivalent else b%4+1),root[1],root[2]+step,root[3],0.,float(0 if equivalent else b%4),0.]
            emit('returned_transition',phase='plan',step=step,command=raw,state=state,reward=-m.metrics(goal,state)['native_distance'],
                returned_goal_state=goal,returned_goal_rgb_sha256='8'*64,terminated=False,truncated=stopped and step==n,**where)
            capture('plan',step)
        row.update(final_state=state.copy(),final_metrics=old_metric(goal,state));physical['rows'].append(row)
        simple('environment_close');emit('branch_closed',receipt=row)
    emit('run_finished',status=physical['status'],counts=physical['counts'],errors=[])
    physical.update(journal_chain_tail=events[-1]['event_sha256'],journal_sha256='7'*64)
    sr=dict(resources={k:dict(attempted=v,completed=v) for k,v in m.scheduler.MAXIMA.items()},scope_resources=scopes,
        limits=dict(elapsed_seconds=10.),actual_backward_observation='mock raw gradient hook count, engine internals unknown',
        prepared='/RAM/scheduler_prepare.json',prepare_sha256='6'*64,artifact_sha256={'panel.json.gz':'5'*64})
    sp=dict(protocol=m.scheduler.protocol(),encoder_receipt=dict(resources=dict(encoder=8,projector=8)),
        historical_anchor_receipts=[dict(status='MOCK_not_actual_model_work')],bindings_sha256={})
    panel=dict(allocations=alloc,signals=[dict(signal=dict(w=1)),dict(signal=dict(w=0))])
    return receipt,sr,sp,panel,physical,pp,events


@pytest.fixture(scope='module')
def full():return fixture()


def test_full_fixed8_model_and_physics_separate(full):
    report=m.assemble_saved(*full)
    assert len(report['rows'])==8 and report['status']=='saved_comparison_assembled_not_admitted'
    assert report['resources']['scheduler_actual']['predict']==dict(attempted=520,completed=520)
    assert report['resources']['physical_actual']=={k:dict(attempted=v,completed=v) for k,v in dict(
        reset=16,public_step=160,explicit_render=176,goal_render=16,set_state=32,physics_step=1632,goal_bind=8,environment_construct=8,environment_close=8).items()}
    assert all(report[k] is False for k in m.FLAGS)
    assert report['RGB_status']['actual_same_Q'] is None
    for row in report['rows']:
        assert row['actual_internal_physics']=={p:dict(attempted=v,completed=v) for p,v in dict(reset=4,warmup=100,plan=100).items()}
        assert row['actual_public_steps']=={p:dict(attempted=10,completed=10) for p in ('warmup','plan')}


def test_stable_join_not_array_order(full):
    original=m.assemble_saved(*full);args=deepcopy(full)
    args[0]['plans'].reverse();args[4]['rows'].reverse();args[5]['plans'].reverse()
    # run_started must bind the exact actual prepare object, independently of its order.
    args[6][0]['prepare']=deepcopy(args[5])
    changed=m.assemble_saved(*args)
    assert changed['rows']==original['rows'] and changed['descriptive']==original['descriptive']


def test_actual_contract_unscoped_branch_closed_identity(full):
    assert all('branch' not in e and type(e['receipt']['branch_index']) is int for e in full[-1] if e['kind']=='branch_closed')
    result=m.assemble_saved(*full)
    assert all(r['independent_physical_dispatches']==4 for r in result['policy_equivalence'])


def test_quota_collision_same_commands_still_eight_physical_dispatches():
    result=m.assemble_saved(*fixture(equivalent=True))
    assert all(e['same_literal_commands'] and e['same_model_Q'] and e['same_actual_final_state'] for e in result['policy_equivalence'])
    assert result['resources']['physical_actual']['public_step']==dict(attempted=160,completed=160)
    for value in result['descriptive']['D_minus_reference'].values():
        assert all(v==0 for v in value['means'].values())
    assert not result['P3_admission'] and 'not evidence of missed dispatch' in result['interpretation']['quota_collision']


@pytest.mark.parametrize('fault',['root','task','blind','boolslot','arm','duplicate','missing','command','selection','native_metric','counts','modelcounts','guard','checkpoint','raw','swapcost'])
def test_mismatches_fail_closed(full,fault):
    args=deepcopy(full);receipt,sr,sp,panel,physical,pp,events=args
    if fault in ('root','task','blind'):
        k=dict(root='root_sha256',task='task_id',blind='blind_sha256')[fault];receipt['plans'][0][k]='f'*64
    elif fault=='boolslot':receipt['plans'][0]['source_slot']=False
    elif fault=='arm':receipt['plans'][0]['arm']='oldGN'
    elif fault=='duplicate':receipt['plans'].append(deepcopy(receipt['plans'][0]))
    elif fault=='missing':physical['rows'].pop()
    elif fault=='command':receipt['plans'][0]['commands'][0][0]=.1
    elif fault=='selection':receipt['plans'][0]['selection_sha256']='f'*64
    elif fault=='native_metric':physical['rows'][0]['final_metrics']['native_distance']+=1
    elif fault=='counts':physical['counts']['public_step']['completed']-=1
    elif fault=='modelcounts':sr['resources']['predict']['completed']=519
    elif fault=='guard':next(e for e in events if e['kind']=='guard_receipt')['accepted']=False
    elif fault=='checkpoint':physical['rows'][0]['checkpoints'][-1]['step']=9
    elif fault=='raw':next(e for e in events if e['kind']=='operation_attempted' and e['resource']=='public_step')['raw_action']=[0.,0.]
    else:receipt['plans'][0]['model_cost']-=1
    with pytest.raises((ValueError,KeyError)):m.assemble_saved(*args)


def test_legal_native_stop_not_fabricated_fullH2():
    report=m.assemble_saved(*fixture(early=True))
    assert report['fixed_branch_denominator']==8
    assert report['descriptive']['full10_paired_summary']=='not_available_do_not_mix_native_stop_with_full10'
    row=next(r for r in report['rows'] if r['source_slot']==0 and r['arm']==m.METHODS[1])
    assert row['native_steps']==5 and not row['full_horizon'] and [c['step'] for c in row['checkpoint_references']]==[0,5]
    assert row['actual_internal_physics']['plan']==dict(attempted=50,completed=50)


@pytest.mark.parametrize('field,value',[('status','partial_or_failed'),('errors',[dict(stage='failed')]),('new_neural_calls',1),('fixed_branch_denominator',7)])
def test_partial_physical_not_a_scientific_negative(full,field,value):
    args=deepcopy(full);args[4][field]=value
    with pytest.raises(ValueError):m.assemble_saved(*args)


def test_native_reward_velocity_is_not_success_threshold():
    goal=np.zeros(7);cur=np.zeros(7);cur[5:]=[1e3,-1e3]
    out=m.metrics(goal,cur)
    assert out['native_success'] and out['wrapped_allxy_success'] and out['native_distance']>1000
    assert out['remaining_native_state_squared_error']==2e6


@pytest.mark.parametrize('turns',[-4,-2,-1,0,1,2,4])
def test_wrapped_angle_invariance_native_expression_not_silently_fixed(turns):
    goal=np.zeros(7);cur=np.zeros(7);cur[4]=.7+turns*2*np.pi
    out=m.metrics(goal,cur)
    assert out['wrapped_object_angle_error']==pytest.approx(.7,abs=1e-14)
    assert out['wrapped_allxy_success'] is False
    if abs(cur[4])>2*np.pi:assert out['negative_native_angle'] and out['native_wrapped_success_disagree']


def test_descriptive_changes_do_not_call_model_Q_actual_gain(full):
    r=m.assemble_saved(*full);d=r['descriptive']['D_minus_reference'][m.METHODS[0]]
    assert d['per_root'][0]['model_Q_D_minus_reference']<0
    assert 'actual_native_distance_D_minus_reference' in d['per_root'][0]
    assert d['actual_same_Q_difference']['status']=='missing_not_encoded'
    assert r['resources']['standalone_logical_not_actual_extra']==dict(zip(m.METHODS,[136,136,136,128]))


def test_cli_help_and_RAM_output(monkeypatch,capsys):
    monkeypatch.setattr(sys,'argv',[m.NAME,'--help'])
    with pytest.raises(SystemExit) as e:m.main()
    assert e.value.code==0 and '--physical-source-sha' in capsys.readouterr().out
    monkeypatch.setattr(m,'inputs',lambda **args:dict(status='RAM_mock_not_actual'))
    monkeypatch.setattr(m,'write_new',lambda output,result:dict(output=output,status=result['status']))
    monkeypatch.setattr(sys,'argv',[m.NAME,'--scheduler-output','/RAM/scheduler','--scheduler-report-sha','a'*64,
        '--physical-output','/RAM/physical','--physical-report-sha','b'*64,'--physical-source-sha','c'*64,'--output','/RAM/new_not_created'])
    m.main();assert 'RAM_mock_not_actual' in capsys.readouterr().out


def test_output_exclusive_floor_and_cap_without_files(monkeypatch):
    monkeypatch.setattr(Path,'exists',lambda self:True)
    with pytest.raises(ValueError,match='NEW16MiB'):m.write_new('/RAM/existing',dict(status='mock'))
    monkeypatch.setattr(Path,'exists',lambda self:False)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda path:SimpleNamespace(free=m.FLOOR+m.CAP-1))
    with pytest.raises(ValueError,match='200MiB'):m.write_new('/RAM/no_create',dict(status='mock'))


def test_size_RAM_only(full):
    r=m.assemble_saved(*full)
    assert len(m.canonical(r))<150*1024


def test_actual_input_adapter_in_RAM_no_import_physical(full,monkeypatch):
    receipt,sr,sp,panel,pr,pp,events=deepcopy(full);sout=Path('/RAM/scheduler');pout=Path('/RAM/physical');psource=m.ROOT/'experiments/spectral_reachability'/(m.PHYSICAL+'.py')
    srh='c'*64;prh='f'*64;psha='1'*64
    pp['bindings']={str(psource):psha};events[0]['prepare']=deepcopy(pp)
    # Re-chain this purely synthetic material after adding its external source pin.
    previous=None
    for e in events:
        e['previous_sha256']=previous;e['event_sha256']=m.digest({k:v for k,v in e.items() if k!='event_sha256'});previous=e['event_sha256']
    pr['journal_chain_tail']=previous
    manifest=dict(schema=m.PHYSICAL+'.selection',arms=list(m.METHODS),selection_fixed_before_physics=True,physical_outcome_selection=False,
        authorities=[dict(authority=m.scheduler.NAME,output=str(sout),report_sha256=srh)])
    binding=dict(prepared=pr['prepared'],prepare_sha256=pr['prepare_sha256'])
    values={str(sout/'report.json'):sr,str(sout/'panel.json.gz'):panel,sr['prepared']:sp,
        str(pout/'report.json'):pr,pr['prepared']:pp,pp['manifest']:manifest,str(pout/'execution_binding.json'):binding}
    hashes={str(sout/'report.json'):srh,str(sout/'panel.json.gz'):'5'*64,sr['prepared']:'6'*64,
        str(pout/'report.json'):prh,pr['prepared']:'e'*64,pp['manifest']:'d'*64,str(pout/'execution_binding.json'):'2'*64,
        str(pout/'journal.jsonl'):pr['journal_sha256'],str(psource):psha,str(Path(m.scheduler.__file__)):m.SCHEDULER_SHA}
    monkeypatch.setattr(m,'sha',lambda path:hashes.get(str(path),'a'*64))
    monkeypatch.setattr(m,'read',lambda path,*args:deepcopy(values[str(path)]))
    monkeypatch.setattr(m.scheduler,'consume',lambda *args:deepcopy(receipt))
    monkeypatch.setattr(Path,'is_symlink',lambda self:False)
    real_stat=Path.stat
    monkeypatch.setattr(Path,'stat',lambda self,*args,**kw:SimpleNamespace(st_size=100) if str(self)==str(pout/'journal.jsonl') else real_stat(self,*args,**kw))
    real_open=Path.open
    monkeypatch.setattr(Path,'open',lambda self,*args,**kw:io.BytesIO(b''.join(m.canonical(e) for e in events)) if str(self)==str(pout/'journal.jsonl') else real_open(self,*args,**kw))
    result=m.inputs(str(sout),srh,str(pout),prh,psha)
    assert len(result['rows'])==8 and result['inputs']['physical_report_sha256']==prh
    assert result['resources']['new_comparison_neural_calls']==result['resources']['new_comparison_physics_calls']==0
