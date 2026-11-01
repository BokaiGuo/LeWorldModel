"""Synthetic saved-native ABI and arithmetic only; no native/NN/fullstrict."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_commoncontroller_physical_comparison_20261029_v1 as m


def h(x):return hashlib.sha256(str(x).encode()).hexdigest()


def fixture(stop=None):
    rp=Path('/RAM/report.json');pp=Path('/RAM/prepare.json');rh,ph=h('report'),h('prepare')
    authority=dict(receipt=dict(predict_API=24,unknown_FLOPs=None))
    report=dict(schema=m.PRODUCER+'.report',status='commoncontroller4_full15_native_recorded_not_admitted',
        fixed_denominator=6,executed_slots=list(m.ACTIVE),missing_slots=[2,5],known_roots=2,fresh_roots=0,inherited=0,
        errors=[],inputs_final_verified=True,new_neural_calls=0,new_independent_samples=0,
        execution_binding=dict(prepared=str(pp),prepare_sha256=ph,planner_authority=authority),
        rows=[],counts={},wall_seconds=1.,**{k:False for k in m.FLAGS})
    for i in range(6):
        root=i//3;goal=[100.,100.,100.,100.,0.,0.,0.];start=[0.,0.,0.,0.,0.,0.,0.]
        commands=np.full((15,2),np.float32(i/16),np.float32)
        plan=dict(index=i,root_slot=root,arm=m.ARMS[i%3],task_id=h(('task',root)),task_sha256=h(('taskfile',root)),
            blind_sha256=h(('blind',root)),native_goal=goal,native_root_sha256=h(('root',root)),
            warm_raw10_bits=h('warm'),history_pixel_sha256=[h(n) for n in range(3)],goal_pixel_sha256=h('goal'),
            commands=commands.tolist(),raw_float32_bits=commands.tobytes().hex(),
            raw_float32_sha256=hashlib.sha256(commands.tobytes()).hexdigest(),selection_digest=h(('seal',i)))
        row=dict(index=i,root_slot=root,condition='off',arm=m.ARMS[i%3],task_id=plan['task_id'],plan=plan,
            selection_digest=plan['selection_digest'],status='missing',checkpoints=[],missing_checkpoints=list(m.CP),
            final_state=None,final_metrics=None,native_success=None,terminated=None,truncated=None,
            initial_history_guard=None,own_attempted=0,own_returned=0,warmup_attempted=0,warmup_returned=0,PNG=0)
        if i not in m.ACTIVE:row['plan']['commands']=None;report['rows'].append(row);continue
        n=8 if stop==i else 15
        responses=[];cps=[]
        for step in range(n+1):
            state=[float(step*(1+i%3)),0.,0.,0.,0.,0.,0.]
            metrics=m.metrics(state,goal)
            if step in m.CP:cps.append(dict(step=step,state=state,snapshot=dict(state=state,bodies=[dict(mock_only=True)]),
                png=f'b{i}_{step}.png',file_sha256=h(('png',root,step)),pixel_sha256=h(('pixel',root,step))))
            if step:responses.append(dict(step=step,state=state,native_metrics=deepcopy(metrics['native']),
                reward=metrics['native7_reward'],native_success=metrics['native_success'],terminated=False,
                truncated=bool(step==n and stop==i)))
        last=responses[-1]
        row.update(status='native_stopped' if stop==i else 'horizon_completed',checkpoints=cps,
            missing_checkpoints=[s for s in m.CP if s>n],own_returned=n,own_attempted=n,warmup_returned=10,warmup_attempted=10,
            PNG=3*(11+n+1),initial_exact=True,initial_history_guard=dict(RAM_only=True),root_state=start,goal=goal,
            responses=responses,final_state=deepcopy(last['state']),final_metrics=deepcopy(last['native_metrics']),
            native_success=False,terminated=False,truncated=last['truncated'],first_error=None,secondary_errors=[])
        report['rows'].append(row)
    public=sum(x['own_returned']+x['warmup_returned'] for x in report['rows'])
    png=sum(x['PNG'] for x in report['rows'])
    report['counts']={k:dict(attempted=v,completed=v) for k,v in dict(constructor=4,close=4,reset=8,set_state=16,
        goal_render=8,goal_bind=4,goal_setter=12,public_step=public,physics_step=16+10*public,explicit_render=png).items()}
    strict=dict(schema=m.PRODUCER+'.consume',status=report['status'],rows=deepcopy(report['rows']),fixed_denominator=6,
        known_roots=2,fresh_roots=0,inherited=0,executed_slots=list(m.ACTIVE),missing_slots=[2,5],errors=[],
        new_neural_calls=0,new_physics_calls=0,physical_report_sha256=rh,counts=deepcopy(report['counts']),
        planner_authority=deepcopy(authority),saved_PNG_verified=png,
        bindings={str(rp):rh,str(pp):ph,str(m.RUNNER):m.RUNNER_SHA,str(m.PROTOCOL):m.PROTOCOL_SHA},
        metric_breakdown=[dict(root_slot=i//3,arm=m.ARMS[i%3],metrics=None if i not in m.ACTIVE else
            m.breakdown(m.metrics(row['final_state'],row['goal']))) for i,row in enumerate(report['rows'])],
        **{k:False for k in m.FLAGS})
    return report,strict,rp,rh,pp,ph


def validate(f):return m.validate(*f)


def test_full_fixed_30_rows_10_pairs_and_difference_directions():
    f=fixture();validate(f);s=m.panel(f[0])
    assert len(s['rows'])==30 and len(s['pairs'])==10
    assert sum(x['metrics'] is None for x in s['rows'])==10
    for pair in s['pairs']:
        assert pair['comparable'] and pair['same_control_time']
        if pair['time']==0:assert all(v==0 for v in pair['delta_program_minus_prior'].values())
        else:
            assert pair['delta_program_minus_prior']['joint_xy_error']<0
            assert pair['delta_program_minus_prior']['native7_reward']>0
            assert pair['program_error_reduction_vs_prior']['joint_xy_error']>0
    assert not s['P2_admission'] and s['paid_replay_receipt']['unknown_FLOPs'] is None
    assert all(abs(p['squared_margin_decomposition']['identity_residual'])<=p['squared_margin_decomposition']['rounding_bound']
        for p in s['pairs'])


def test_earlystop_null_checkpoint_but_true_terminal_step_preserved():
    f=fixture(stop=1);validate(f);s=m.panel(f[0])
    r=next(x for x in s['rows'] if x['slot']==1 and x['time']==15)
    assert r['metrics'] is None and r['truncated'] is None
    end=next(x for x in s['pairs'] if x['root_slot']==0 and x['time']=='native_terminal')
    assert end['program_actual_plan_step']==8 and end['prior_actual_plan_step']==15
    assert end['comparable'] and not end['same_control_time']
    cp=next(x for x in s['pairs'] if x['root_slot']==0 and x['time']==15)
    assert not cp['comparable'] and cp['delta_program_minus_prior'] is None and cp['missing_slots']==[1]
    assert cp['squared_margin_decomposition'] is None


@pytest.mark.parametrize('kind',['denominator','active','rows','metric_missing','metric_identity','metric_number','authority',
    'reportsha','bindings','source','flags','count','missing','cp','response','raw','termination'])
def test_complete_receipt_and_numeric_mutations_not_status_only(kind):
    f=list(deepcopy(fixture()));r,s=f[:2]
    if kind=='denominator':s['fixed_denominator']=4
    elif kind=='active':s['executed_slots']=[0,1,2,3]
    elif kind=='rows':s['rows'][0]['root_slot']=1
    elif kind=='metric_missing':s['metric_breakdown'].pop()
    elif kind=='metric_identity':s['metric_breakdown'][0]['arm']='program_first_postcondition'
    elif kind=='metric_number':s['metric_breakdown'][0]['metrics']['velocity_error']=1.
    elif kind=='authority':s['planner_authority']['receipt']['predict_API']=23
    elif kind=='reportsha':s['physical_report_sha256']=h('bad')
    elif kind=='bindings':s['bindings'][str(f[4])]=h('bad')
    elif kind=='source':s['bindings'][str(m.RUNNER)]=h('bad')
    elif kind=='flags':s['freshness']=True
    elif kind=='count':r['counts']['physics_step']['completed']-=1;s['counts']=deepcopy(r['counts'])
    else:
        row=r['rows'][0]
        if kind=='missing':r['rows'][2]['final_state']=[0.]*7
        elif kind=='cp':row['checkpoints'][1]['snapshot']['state']=[9.]*7
        elif kind=='response':row['responses'][0]['reward']=0.
        elif kind=='raw':row['plan']['commands'][0][0]=.9
        elif kind=='termination':row['responses'][0]['terminated']=True
        s['rows']=deepcopy(r['rows'])
    with pytest.raises((ValueError,KeyError)):validate(f)


def test_cp0_full_snapshot_control_and_same_task_not_state_only():
    f=fixture();r=f[0]
    for mode in ('snapshot','task'):
        q=deepcopy(r)
        if mode=='snapshot':q['rows'][1]['checkpoints'][0]['snapshot']['bodies'][0]['mock_only']=False
        else:q['rows'][1]['plan']['blind_sha256']=h('other')
        with pytest.raises(ValueError):m.panel(q)


def test_independent_float64_axes_success_velocity_raw_and_wrapped_angles():
    s=[3.,4.,0.,0.,0.,12.,0.];g=[0.]*7;x=m.metrics(s,g)
    assert x['joint_xy_squared']==25. and x['velocity_squared']==144.
    assert x['native7_distance']==13. and x['native7_squared']==169. and x['native7_reward']==-13.
    assert x['native_success']  # velocity excluded from original success.
    y=m.metrics([0.,0.,0.,0.,4*math.pi,0.,0.],g)
    assert y['negative_native_angle'] and y['native_angle_term']==-2*math.pi
    assert y['raw_angle_squared']==(4*math.pi)**2
    assert abs(y['wrapped_angle_error'])<1e-14
    assert m.metrics([20.,0.,0.,0.,0.,0.,0.],g)['native_success'] is False


def test_position_improves_but_velocity_harms_native7_no_single_cause_claim():
    goal=[0.]*7
    prior=m.metrics([30.,0.,0.,0.,0.,0.,0.],goal)
    program=m.metrics([10.,0.,0.,0.,0.,40.,0.],goal)
    assert program['joint_xy_squared']-prior['joint_xy_squared']==-800.
    assert program['velocity_squared']-prior['velocity_squared']==1600.
    assert program['native7_squared']-prior['native7_squared']==800.
    assert program['native7_distance']>prior['native7_distance']
    assert program['native_success'] and not prior['native_success']
    for x in (prior,program):
        d=x['native7_squared_decomposition']
        assert d['native7_squared']==d['sum_of_three_axes']
        assert d['identity_residual']==0.


def test_scalar_bool_reject_and_signedzero_raw_bits():
    with pytest.raises(ValueError):m.metrics([True]+[0.]*6,[0.]*7)
    r=fixture()[0];p=r['rows'][0]['plan'];p['commands'][0][0]=-0.
    with pytest.raises(ValueError):m.rawbits(p)


def test_bound_read_and_exclusive_write_fixture(tmp_path):
    p=tmp_path/'input.json';p.write_bytes(m.encode(dict(RAM_only=True)))
    assert m.read_bound(str(p),m.sha(p))==dict(RAM_only=True)
    with pytest.raises(ValueError):m.read_bound(str(p),h('bad'))
    output=tmp_path/(m.NAME+'_RAM.json');m.write_new(str(output),dict(RAM_only=True))
    with pytest.raises(ValueError):m.write_new(str(output),dict(RAM_only=True))


def test_analyze_verify_cli_fixture_without_producer_or_fullstrict(tmp_path,monkeypatch):
    r,s,rp,rh,pp,ph=fixture();pp=tmp_path/'prepare.json';rp=tmp_path/'report.json';sp=tmp_path/'receipt.json'
    p=dict(schema=m.PRODUCER+'.prepare',plans=[x['plan'] for x in r['rows']],
        source_report_sha256=h('source'),planner_authority=deepcopy(r['execution_binding']['planner_authority']),
        source_roots=[dict(goal=r['rows'][i]['goal'],root_state=r['rows'][i]['root_state'],
            root_sha256=r['rows'][i]['plan']['native_root_sha256']) for i in (0,3)])
    pp.write_bytes(m.encode(p));ph=m.sha(pp)
    r['execution_binding'].update(prepared=str(pp),prepare_sha256=ph,source_report_sha256=h('source'))
    rp.write_bytes(m.encode(r));rh=m.sha(rp)
    s['physical_report_sha256']=rh;s['bindings']={str(rp):rh,str(pp):ph,str(m.RUNNER):m.RUNNER_SHA,str(m.PROTOCOL):m.PROTOCOL_SHA}
    sp.write_bytes(m.encode(s));sh=m.sha(sp)
    # Only own-file hashes need real doc/source; producer imports forbidden below.
    import builtins
    original=builtins.__import__
    def guarded(name,*a,**kw):
        assert not any(x in name for x in ('commoncontroller_physical_20261029_v1','torch','gymnasium','pymunk'))
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    output=tmp_path/(m.NAME+'_RAM.json')
    m.main(['analyze','--report',str(rp),'--report-sha',rh,'--receipt',str(sp),'--receipt-sha',sh,'--output',str(output)])
    m.main(['verify','--summary',str(output),'--summary-sha',m.sha(output)])


def test_cold_import_no_runner_or_neural():
    code="import sys;from experiments.spectral_reachability import agent_p2_commoncontroller_physical_comparison_20261029_v1;assert not any(k in sys.modules for k in ('torch','gymnasium','pymunk','experiments.spectral_reachability.agent_p2_commoncontroller_physical_20261029_v1'))"
    subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,check=True,capture_output=True)


def test_actual_saved_schema_identity_only_no_scalar_analyze_or_fullstrict():
    rp=m.ROOT/'results/agent_p2_commoncontroller_physical_20261029_v1_MAIN01/report.json'
    sp=m.ROOT/'results/agent_p2_commoncontroller_physical_20261029_v1_consume_MAIN01.json'
    r=m.read_bound(str(rp),'3164f9fb3808e6d579ae9aada8306427b0acbf9a657eda2f2e7e46f68f4f5076')
    s=m.read_bound(str(sp),'bf976c4f28f3b4634d2d78594779cc74933bf8870d78aef2db8cb182a49342b3')
    synthetic=fixture()[1]
    assert set(synthetic)<=set(s)
    assert s['schema']==m.PRODUCER+'.consume'
    assert s['fixed_denominator']==len(s['rows'])==len(s['metric_breakdown'])==6
    assert s['bindings'][str(rp)]==s['physical_report_sha256']
    assert m.encode(r['rows'])==m.encode(s['rows'])
    assert set(s['metric_breakdown'][0]['metrics'])==set(synthetic['metric_breakdown'][0]['metrics'])
    # Deliberately do not call validate/panel/analyze on these actual outcomes.


@pytest.mark.parametrize('field',['boolcount','boolactive','droppedcp'])
def test_narrow_type_and_checkpoint_completeness(field):
    f=list(deepcopy(fixture()));r,s=f[:2]
    if field=='boolcount':r['counts']['constructor']['completed']=True;s['counts']=deepcopy(r['counts'])
    elif field=='boolactive':s['executed_slots'][0]=False
    else:
        r['rows'][0]['checkpoints'].pop();r['rows'][0]['missing_checkpoints']=[15];s['rows']=deepcopy(r['rows'])
    with pytest.raises(ValueError):validate(f)
