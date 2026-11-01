"""Synthetic saved ABI only: no native driver, fullstrict, model or environment."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_six_target_physical_scalar_summary_20261029_v1 as m


def h(x):return hashlib.sha256(str(x).encode()).hexdigest()


def native(s,g):
    d=[x-y for x,y in zip(s,g)];xy=math.sqrt(sum(x*x for x in d[:4]));angle=min(abs(d[4]),2*math.pi-abs(d[4]))
    wrapped=abs(math.atan2(math.sin(d[4]),math.cos(d[4])))
    return dict(native_distance=math.sqrt(sum(x*x for x in d)),native_success=xy<20 and angle<math.pi/9,
        native_angle_term=angle,negative_native_angle=angle<0,object_pose=s[2:5],object_position_error=math.sqrt(sum(x*x for x in d[2:4])),
        object_success=None,wrapped_allxy_success=xy<20 and wrapped<math.pi/9,wrapped_object_angle_error=wrapped)


def fixture(stop=None,warmstop=None):
    paths=[Path('/RAM/prepare.json'),Path('/RAM/report.json'),Path('/RAM/strict.json')];hashes=[h(x) for x in paths]
    p=dict(schema=m.PRODUCER+'.prepare',source_report_sha256=h('source'),planner_authority=dict(receipt=dict(FLOPs=None)),
           plans=[],source_roots=[])
    for root in (0,1):p['source_roots'].append(dict(root_state=[0.]*7,goal=[100.,100.,100.,100.,0.,0.,0.],root_sha256=h(root)))
    r=dict(schema=m.PRODUCER+'.report',status=m.STATUS,fixed_denominator=6,executed_slots=list(range(6)),missing_slots=[],
        known_roots=2,fresh_roots=0,inherited=0,errors=[],inputs_final_verified=True,new_neural_calls=0,new_independent_samples=0,
        execution_binding=dict(prepared=str(paths[0]),prepare_sha256=hashes[0],source_report_sha256=p['source_report_sha256'],
            planner_authority=deepcopy(p['planner_authority'])),rows=[],counts={},wall_seconds=1.,**{k:False for k in m.FLAGS})
    for i in range(6):
        root=p['source_roots'][i//3];a=np.full((15,2),np.float32((i%3)/4),np.float32)
        plan=dict(index=i,root_slot=i//3,condition='off',arm=m.ARMS[i%3],task_id=h(('task',i//3)),
            task_sha256=h(('taskfile',i//3)),blind_sha256=h(('blind',i//3)),native_goal=deepcopy(root['goal']),
            native_root_sha256=root['root_sha256'],warm_raw10_bits='warm',history_pixel_sha256=[h(x) for x in range(3)],
            goal_pixel_sha256=h('goal'),selection_digest=h(('selection',i)),commands=a.tolist(),raw_float32_bits=a.tobytes().hex(),
            raw_float32_sha256=hashlib.sha256(a.tobytes()).hexdigest())
        p['plans'].append(deepcopy(plan));n=0 if warmstop==i else (8 if stop==i else 15);w=5 if warmstop==i else 10
        row=dict(index=i,root_slot=i//3,condition='off',arm=m.ARMS[i%3],task_id=plan['task_id'],plan=plan,
            selection_digest=plan['selection_digest'],status='native_stopped' if i in (stop,warmstop) else 'horizon_completed',
            first_error=None,secondary_errors=[],own_returned=n,own_attempted=n,warmup_returned=w,warmup_attempted=w,
            initial_exact=warmstop!=i,initial_history_guard=None if warmstop==i else dict(RAM_only=True),
            root_state=deepcopy(root['root_state']),goal=deepcopy(root['goal']),responses=[],checkpoints=[],trajectory=[],
            final_state=None,final_metrics=None,native_success=None,terminated=None,truncated=None,PNG=3*(w+1+(n+1 if warmstop!=i else 0)))
        for step in range(n+1) if warmstop!=i else ():
            s=[float(step*(1+i%3)),0.,0.,0.,0.,float(step*(i%3)),0.];nm=native(s,row['goal'])
            if step:
                row['responses'].append(dict(step=step,state=deepcopy(s),native_metrics=nm,reward=-nm['native_distance'],
                    native_success=nm['native_success'],terminated=nm['native_success'],truncated=step==n and stop==i))
            if step in m.CP:
                f=dict(step=step,state=deepcopy(s),snapshot=dict(state=deepcopy(s),other='full'),png=f'b{i}_{step}.png',
                    file_sha256=h((i//3,step)),pixel_sha256=h(('pixel',i//3,step)))
                row['checkpoints'].append(deepcopy(f));row['trajectory'].append(deepcopy(f))
        row['missing_checkpoints']=[s for s in m.CP if s>n or warmstop==i]
        if n:
            last=row['responses'][-1];row.update(final_state=deepcopy(last['state']),final_metrics=deepcopy(last['native_metrics']),
                **{k:last[k] for k in ('native_success','terminated','truncated')})
        r['rows'].append(row)
    pub=sum(x['warmup_returned']+x['own_returned'] for x in r['rows']);png=sum(x['PNG'] for x in r['rows']);goals=sum(x['initial_exact'] for x in r['rows'])
    r['counts']={k:dict(attempted=v,completed=v) for k,v in dict(constructor=6,close=6,reset=12,set_state=24,goal_render=12,
        goal_setter=12+goals,goal_bind=goals,public_step=pub,physics_step=24+10*pub,explicit_render=png).items()}
    c={k:deepcopy(r[k]) for k in ('status','fixed_denominator','executed_slots','missing_slots','known_roots','fresh_roots','inherited','errors','rows','counts',*m.FLAGS)}
    c.update(schema=m.PRODUCER+'.consume',new_neural_calls=0,new_physics_calls=0,physical_report_sha256=hashes[1],
        planner_authority=deepcopy(p['planner_authority']),saved_PNG_verified=png,wall_through_report_fsync_hash=2.,
        bindings={str(paths[0]):hashes[0],str(paths[1]):hashes[1],str(m.RUNNER):m.RUNNER_SHA,str(m.PROTOCOL):m.PROTOCOL_SHA},
        metric_breakdown=[dict(root_slot=i//3,arm=m.ARMS[i%3],metrics=None if b['final_state'] is None else dict(
            native=deepcopy(b['final_metrics']),native_success=b['native_success'],joint_xy_error=math.dist(b['final_state'][:4],b['goal'][:4]),
            angle_native_term=0.,angle_wrapped_error=0.,velocity_error=abs(b['final_state'][5]),
            native7_distance=b['final_metrics']['native_distance'],native7_reward=-b['final_metrics']['native_distance'])) for i,b in enumerate(r['rows'])])
    return p,r,c,paths,hashes


def test_full_six_native_shape_all_24_cp_pairs_and_resources():
    f=fixture();m.validate(*f);s=m.panel(f[1])
    assert s['denominator']==dict(slots=6,checkpoints=24,pairs=24) and s['comparable_pairs']==24
    assert len(s['native_terminals'])==6 and s['physical_counts']['reset']['completed']==12
    assert s['physical_counts']['physics_step']['completed']==1524
    for pair in s['pairs']:
        if pair['step']==0:assert set(pair['directions'].values())=={'tie'}
        else:assert pair['delta_right_minus_left']['joint_xy_error']<0 and pair['delta_right_minus_left']['agent_velocity_squared']>0
    assert all(s[k] is False for k in m.FLAGS)


@pytest.mark.parametrize('stop,warmstop',[(1,None),(None,5)])
def test_native_earlystop_null_no_cp15_fill(stop,warmstop):
    f=fixture(stop,warmstop);m.validate(*f);s=m.panel(f[1]);slot=stop if stop is not None else warmstop
    assert len(s['rows'])==len(s['pairs'])==24
    assert next(x for x in s['rows'] if x['slot']==slot and x['step']==15)['metrics'] is None
    assert any(x['delta_right_minus_left'] is None for x in s['pairs'])
    assert s['native_terminals'][slot]['actual_step']==(8 if stop is not None else None)


@pytest.mark.parametrize('kind',['schema','subset','source','prepare','report','authority','rows','metric','reset','public',
    'raw','signedzero','task','snapshot','response','missing','terminal','flag','PNG'])
def test_mutation_rejection(kind):
    p,r,c,paths,hs=deepcopy(fixture())
    if kind=='schema':c['schema']='old.consume'
    elif kind=='subset':c['executed_slots']=[0,1,3,4]
    elif kind=='source':c['bindings'][str(m.RUNNER)]=h('bad')
    elif kind=='prepare':r['execution_binding']['prepare_sha256']=h('bad')
    elif kind=='report':c['physical_report_sha256']=h('bad')
    elif kind=='authority':c['planner_authority']['receipt']['FLOPs']=0
    elif kind=='rows':c['rows'][0]['index']=9
    elif kind=='metric':c['metric_breakdown'][0]['metrics']['velocity_error']=2.
    elif kind in ('reset','public'):
        r['counts']['reset' if kind=='reset' else 'public_step']['completed']-=1;c['counts']=deepcopy(r['counts'])
    elif kind=='flag':c['P2_admission']=True
    elif kind=='PNG':c['saved_PNG_verified']-=1
    else:
        row=r['rows'][0]
        if kind=='raw':row['plan']['commands'][0][0]=.5
        elif kind=='signedzero':row['plan']['commands'][0][0]=-0.
        elif kind=='task':row['task_id']='bad'
        elif kind=='snapshot':row['checkpoints'][1]['snapshot']['state'][0]+=1
        elif kind=='response':row['responses'][0]['reward']=0.
        elif kind=='missing':row['missing_checkpoints']=[15]
        else:row['final_state'][0]+=1
        c['rows']=deepcopy(r['rows'])
    with pytest.raises((ValueError,KeyError)):m.validate(p,r,c,paths,hs)


def test_independent_closed_form_axes_and_native_angle_bug_kept():
    x=m.metrics([3.,4.,0.,0.,0.,0.,12.],[0.]*7)
    assert x['native7_squared']==169 and x['native7_distance']==13
    assert x['joint_xy_squared']==25 and x['agent_velocity_squared']==144
    y=m.metrics([0.,0.,0.,0.,8.,0.,0.],[0.]*7)
    assert y['native_angle_term']<0 and y['native_success'] and not y['wrapped_success_diagnostic']
    assert y['squared_decomposition']['total']==64


def test_cp0_control_requires_full_snapshot_and_same_history():
    for key in ('snapshot','history'):
        r=fixture()[1]
        if key=='snapshot':r['rows'][1]['checkpoints'][0]['snapshot']['other']='changed'
        else:r['rows'][1]['plan']['history_pixel_sha256'][0]=h('changed')
        with pytest.raises(ValueError):m.panel(r)


def test_disk_bound_analysis_and_wholeverify(tmp_path):
    p,r,c,_,_=fixture();pp=tmp_path/'prepare.json';rp=tmp_path/'report.json';sp=tmp_path/'receipt.json'
    pp.write_bytes(m.encode(p));ph=m.sha(pp);r['execution_binding'].update(prepared=str(pp),prepare_sha256=ph)
    rp.write_bytes(m.encode(r));rh=m.sha(rp);c['physical_report_sha256']=rh
    c['bindings'].update({str(pp):ph,str(rp):rh});sp.write_bytes(m.encode(c));ch=m.sha(sp)
    result=m.analyze(str(pp),ph,str(rp),rh,str(sp),ch)
    output=tmp_path/(m.NAME+'_RAM.json');m.write_new(str(output),result)
    m.main(['verify','--summary',str(output),'--summary-sha',m.sha(output)])
    with pytest.raises(ValueError):m.write_new(str(output),result)
    bad=tmp_path/'bad.json';bad.write_bytes(b'{"a":1,"a":2}')
    with pytest.raises(ValueError):m.read_bound(str(bad),m.sha(bad))


def test_cold_import_has_no_producer_math_torch_or_environment():
    code="import sys;from experiments.spectral_reachability import "+m.NAME+";assert not any(k.startswith(('torch','gym','pymunk','stable_worldmodel')) or k.endswith('agent_p2_six_target_physical_20261029_v1') for k in sys.modules)"
    subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,check=True)


def test_literal_vector_and_path_failclosed():
    for x in ([False]*7,[float('nan')]*7,[0.]*6):
        with pytest.raises(ValueError):m.metrics(x,[0.]*7)
    with pytest.raises(ValueError):m.absolute('results/relative.json')


def test_actual_saved_schema_metadata_only_no_analysis_or_fullstrict():
    path=m.ROOT/'results/agent_p2_six_target_physical_20261029_v1_consume_MAIN01.json'
    c=m.read_bound(str(path),'7579693032046b2427a7d5b492a90131029fe87d39147d06fe49123a3cfead13')
    assert c['schema']==m.PRODUCER+'.consume' and len(c['rows'])==len(c['metric_breakdown'])==6
    assert {'physical_report_sha256','bindings','planner_authority','counts','saved_PNG_verified'}<=c.keys()
    for row in c['rows']:
        assert {'plan','responses','trajectory','checkpoints','missing_checkpoints','initial_exact','initial_history_guard'}<=row.keys()
        for f in row['checkpoints']:
            assert set(f)=={'step','png','file_sha256','pixel_sha256','state','snapshot'}
            assert {'state','state_float64_bits','bodies'}<=f['snapshot'].keys()
        for x in row['responses']:
            assert {'state','step','native_metrics','native_success','reward','terminated','truncated'}<=x.keys()
    assert set(c['metric_breakdown'][0]['metrics'])=={'native','native_success','joint_xy_error','angle_native_term',
        'angle_wrapped_error','velocity_error','native7_distance','native7_reward'}
