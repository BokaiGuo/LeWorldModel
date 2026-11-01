"""RAM/temporary files only, never physical or scientific evidence."""
from copy import deepcopy
import json
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_commit_vs_mpc_metrics_20261029_v1 as m


def state(step):
    return [30.0+step,40.0,20.0,30.0,.125,3.0,4.0]


def cp(step, old=False):
    s=state(step)
    out=dict(step=step,state=s,snapshot=dict(state=deepcopy(s),bodies=dict(block=dict(angle=.125))),
        pixel_sha256=f'{step+1:064x}',file_sha256=f'{step+2:064x}')
    if old:out.update(phase='plan',goal=[0.0]*7,path=f'/fixture/cp{step}.png')
    else:out['png']=f'cp{step}.png'
    return out


def fixture():
    oldrows=[];nodes=[];plans=[];rows=[];events=[];ops=[]
    for root in (0,1):
        for arm in ('fixedH1_endpoints','fixedH1_midpoints','adaptive_endpoints','adaptive_midpoints'):
            eid=f's{root}_{arm}'
            commands=[[0.0,0.0] for _ in range(20)]
            start=10 if arm.startswith('adaptive') else 5
            for j in range(start,20):commands[j][0]=.25
            oldrows.append(dict(episode_id=eid,source_slot=root,mode=arm,control_returned=20,executed_raw=commands,
                checkpoints=[cp(b,True) for b in m.CP]))
            for b in (5,10,15):
                nodes.append(dict(episode_id=eid,boundary=b,request=dict(task=dict(goal=[1.]*192),boundary=b),
                    receipt=dict(selected_horizon=3 if arm.startswith('adaptive') else 1,
                        panels=[dict(horizon=1,winner=dict(actions=[[0.,0.]]*5)),
                            dict(horizon=3,winner=dict(index=0,actions=[[0.,0.] for _ in range(15)],whole=[{'context':'fixture'}]*3))])))
    authority=dict(report_sha256=m.OLD_REPORT_SHA,receipt_sha256=m.OLD_STRICT_SHA)
    for i,(root,bank) in enumerate(m.ORDER):
        node=next(n for n in nodes if n['episode_id']==f's{root}_adaptive_{bank}' and n['boundary']==5)
        plan=dict(index=i,root_slot=root,condition='off',arm=bank,task_id=f'fixturetask{root}',selection_digest=f'{i+1:064x}',
            commands=[[0.,0.] for _ in range(20)],prefix_commands=[[0.,0.] for _ in range(5)],suffix_commands=[[0.,0.] for _ in range(15)],
            native_goal=[0.]*7,source_witness=dict(winner=deepcopy(node['receipt']['panels'][1]['winner']),request=deepcopy(node['request'])))
        responses=[]
        for step in range(1,21):
            s=state(step);calc=m.metrics(plan['native_goal'],s)
            e=dict(kind='returned_transition',branch=i,phase='plan',step=step,state=s,goal_state=[0.]*7,
                actual_dtype=np.dtype(np.float32).str,actual_bits=m.action_bits([[0.,0.]],1).hex(),requested_bits=m.action_bits([[0.,0.]],1).hex(),
                reward=calc['native7_reward'],terminated=False,truncated=False)
            events.append(e)
            responses.append(dict(step=step,state=deepcopy(s),reward=e['reward'],terminated=False,truncated=False,native_success=False))
            ops.append(dict(resource='public_step',branch=i,phase='plan',step=step,status='returned',raw_bits=e['actual_bits']))
        row=dict(index=i,root_slot=root,condition='off',arm=bank,task_id=plan['task_id'],selection_digest=plan['selection_digest'],
            plan=deepcopy(plan),own_returned=20,status='horizon_completed',initial_exact=True,
            checkpoints=[cp(b) for b in m.CP],missing_checkpoints=[],responses=responses,
            final_state=state(20),terminated=False,truncated=False,first_error=None,secondary_errors=[])
        plans.append(plan);rows.append(row)
    flags=dict(P1_admission=False,P3_admission=False,C3=False,freshness=False,scientific_admission=False)
    p=dict(schema=m.PHYSICAL+'.prepare',plans=plans,contract={'fixture':'not an actual run'},planner_authority=authority)
    r=dict(schema=m.PHYSICAL+'.report',rows=rows,contract=deepcopy(p['contract']),execution_binding=dict(planner_authority=authority),
        status='h3_witness4_control20_native_recorded_not_admitted',counts=dict(public_step=dict(attempted=120,completed=120)),
        errors=[],actual_operation_receipts=ops,fixed_denominator=4,known_roots=2,fresh_roots=0,**flags)
    receipt=dict(schema=m.PHYSICAL+'.consume',rows=deepcopy(rows),plans=deepcopy(plans),planner_authority=deepcopy(authority),
        counts=deepcopy(r['counts']),errors=[],status=r['status'],fixed_denominator=4,known_roots=2,fresh_roots=0,**flags)
    old=dict(schema=m.SHARED+'.report',rows=oldrows,nodes=nodes,**flags)
    old_receipt=dict(schema=m.SHARED+'.consume',complete=True,report_sha256=m.OLD_REPORT_SHA,
        status='saved_shared_panel_verified_NOT_admitted',**flags)
    return p,r,receipt,old,old_receipt,events


def sync(p,r,s):
    s['rows']=deepcopy(r['rows']);s['plans']=deepcopy(p['plans'])
    s['errors']=deepcopy(r['errors']);s['status']=r['status'];s['counts']=deepcopy(r['counts'])


def truncate(args,index,n):
    p,r,s,old,old_s,events=args;row=r['rows'][index]
    row.update(status='native_stopped',own_returned=n,terminated=False,truncated=True,
        final_state=None if n==0 else state(n))
    row['responses']=row['responses'][:n]
    if n:row['responses'][-1]['truncated']=True
    row['checkpoints']=[c for c in row['checkpoints'] if c['step']<=n]
    row['missing_checkpoints']=[b for b in m.CP if b>n]
    events[:]=[e for e in events if e['branch']!=index or e['step']<=n]
    if n:next(e for e in events if e['branch']==index and e['step']==n)['truncated']=True
    r['actual_operation_receipts']=[o for o in r['actual_operation_receipts'] if o['branch']!=index or o['step']<=n]
    sync(p,r,s)


def test_full_four_primary_auxiliary_and_zero_latent():
    a=fixture();before=deepcopy(a);out=m.summarize(*a)
    assert a==before
    assert len(out['rows'])==len(out['primary_pairs'])==len(out['auxiliary_pairs'])==4
    assert sum(len(p['checkpoints']) for p in out['primary_pairs'])==20
    assert out['latent_metrics'] is None and out['new_neural_calls']==0
    assert all(x['complete_same_action_H3_physical_path'] for x in out['rows'])
    assert all(x['actions']['first_original_MPC_difference']==11 for x in out['rows'])
    assert all(x['effect_interpretable'] for x in out['primary_pairs'])


def test_independent_metric_axes_native_angle_velocity():
    q=m.metrics([0.]*7,[3.,4.,0.,0.,7.,12.,5.])
    assert q['joint_xy']==5 and q['agent_xy']==5 and q['object_xy']==0
    assert q['velocity_error']==13 and q['native7_distance']==np.sqrt(243)
    assert q['native7_reward']==-np.sqrt(243)
    assert q['native_angle_term']==2*np.pi-7 and q['native_success'] is True
    assert q['angle_wrapped']>0 and q['angle_absolute']==7


def test_stop14_no_cp15_20_and_eventtime_not_substituted():
    a=fixture();truncate(a,0,14);out=m.summarize(*a)
    row=out['rows'][0]
    assert row['eventtime_terminal']['step']==14 and row['eventtime_terminal']['truncated'] is True
    assert row['at20'] is None and row['missing_checkpoints']==[15,20]
    assert row['actions']['missing_selected_actions']==6
    assert not row['complete_same_action_H3_physical_path']
    pair=out['primary_pairs'][0]['checkpoints'][-1]
    assert pair['commit_metrics'] is pair['commit_minus_reference'] is None
    assert pair['success']['commit'] is None and pair['reference_metrics'] is not None


def test_early_stop_before_control_not_dropped():
    a=fixture();truncate(a,1,4);out=m.summarize(*a)
    row=out['rows'][1]
    assert row['exact_controls_vs_MPC']['5']['exact'] is None
    assert row['effect_interpretable'] is False
    assert len(out['primary_pairs'])==4


def test_warmup_stop_no_plan_observation_all_missing():
    a=fixture();truncate(a,0,0)
    p,r,s,old,old_s,events=a;row=r['rows'][0]
    row.update(initial_exact=False,checkpoints=[],missing_checkpoints=list(m.CP),terminated=None,truncated=None)
    sync(p,r,s);out=m.summarize(*a)
    assert out['rows'][0]['eventtime_terminal'] is None
    assert out['rows'][0]['at20'] is None
    assert all(c['commit_missing'] for c in out['primary_pairs'][0]['checkpoints'])
    assert out['rows'][0]['effect_interpretable'] is False


def test_native_success_at14_is_not_cp20_success():
    a=fixture();truncate(a,0,14);p,r,s,old,old_s,events=a
    row=r['rows'][0];e=next(e for e in events if e['branch']==0 and e['step']==14)
    e.update(state=[0.]*7,reward=-0.0,terminated=True,truncated=False)
    row['responses'][-1].update(state=[0.]*7,reward=-0.0,native_success=True,terminated=True,truncated=False)
    row.update(final_state=[0.]*7,terminated=True,truncated=False)
    sync(p,r,s);out=m.summarize(*a)
    assert out['rows'][0]['eventtime_terminal']['metrics']['native_success'] is True
    assert out['rows'][0]['at20'] is None
    assert out['primary_pairs'][0]['checkpoints'][-1]['success']['commit'] is None


@pytest.mark.parametrize('key', ['state','snapshot','pixel_sha256'])
@pytest.mark.parametrize('step', [5,10])
def test_exact_control_mismatch_is_engineering_not_science(key,step):
    a=fixture();p,r,s,old,old_s,events=a
    c=next(c for c in r['rows'][0]['checkpoints'] if c['step']==step)
    if key=='state':
        # Mutate both physical labels consistently to isolate cross-run control.
        c['state'][0]+=1;c['snapshot']['state'][0]+=1
        e=next(e for e in events if e['branch']==0 and e['step']==step)
        e['state'][0]+=1;e['reward']=m.metrics([0.]*7,e['state'])['native7_reward']
        response=r['rows'][0]['responses'][step-1];response['state']=deepcopy(e['state']);response['reward']=e['reward']
    elif key=='snapshot':c['snapshot']['bodies']['block']['angle']=-0.0
    else:c[key]='f'*64
    sync(p,r,s);out=m.summarize(*a);row=out['rows'][0]
    assert row['exact_controls_vs_MPC'][str(step)]['exact'] is False
    assert row['control_failure_not_scientific_negative'] is True
    assert not row['effect_interpretable'] and not out['primary_pairs'][0]['effect_interpretable']


def test_cp10_not_control_against_fixedH1():
    a=fixture();old=a[3]
    c=next(c for c in old['rows'][0]['checkpoints'] if c['step']==10)
    c['state'][0]+=1;c['snapshot']['state'][0]+=1
    out=m.summarize(*a)
    assert out['rows'][0]['effect_interpretable'] is True
    assert '10' not in out['rows'][0]['exact_controls_vs_fixedH1']
    delta=out['auxiliary_pairs'][0]['checkpoints'][2]['commit_minus_reference']
    assert delta['joint_xy']<0


@pytest.mark.parametrize('mutate',[
    lambda a:a[1]['rows'].reverse(),
    lambda a:a[1]['rows'][0]['missing_checkpoints'].append(20),
    lambda a:a[1]['rows'][0]['checkpoints'].pop(),
    lambda a:a[0]['plans'][0]['source_witness']['winner']['actions'][2].__setitem__(0,.5),
    lambda a:a[0]['plans'][0]['source_witness']['request'].__setitem__('boundary',10),
    lambda a:a[5][0].__setitem__('actual_bits','ff'*8),
    lambda a:a[5][0].__setitem__('requested_bits','ff'*8),
    lambda a:a[5][0].__setitem__('reward',0.),
    lambda a:a[5][0].__setitem__('terminated',True),
    lambda a:a[5][0].__setitem__('goal_state',[1.]*7),
    lambda a:a[1]['actual_operation_receipts'][0].__setitem__('raw_bits','ff'*8),
    lambda a:a[1]['rows'][0]['responses'][0].__setitem__('native_success',True),
    lambda a:a[1]['rows'][0]['checkpoints'][0]['snapshot']['state'].__setitem__(0,0.),
    lambda a:a[0]['plans'][0]['commands'][0].__setitem__(0,-0.),
])
def test_mutations_fail_closed(mutate):
    a=fixture();mutate(a)
    # Synchronize duplicated strict/plan fields so independent checks are tested.
    for row,plan in zip(a[1]['rows'],a[0]['plans']):
        row['plan']=deepcopy(plan)
    sync(*a[:3])
    with pytest.raises(ValueError):m.summarize(*a)


def test_missing_row_not_allowed_and_success_not_inferred():
    a=fixture();a[2]['rows'].pop()
    with pytest.raises(ValueError):m.summarize(*a)
    a=fixture();a[1]['rows'][0]['own_returned']=19;sync(*a[:3])
    with pytest.raises(ValueError):m.summarize(*a)


def test_real_saved_io_analyze_verify_no_producer(tmp_path,monkeypatch,capsys):
    p,r,s,old,old_s,events=fixture()
    pp=tmp_path/'prepare.json';pp.write_bytes(m.raw(p));ph=m.sha(pp)
    r['execution_binding'].update(prepared=str(pp),prepare_sha256=ph)
    journal=tmp_path/'journal.jsonl';journal.write_bytes(b''.join(m.raw(e)+b'\n' for e in events));r['journal_sha256']=m.sha(journal)
    rp=tmp_path/'report.json';rp.write_bytes(m.raw(r));rh=m.sha(rp)
    s.update(physical_report_sha256=rh,bindings={str(pp):ph,str(rp):rh,str(journal):m.sha(journal)})
    sp=tmp_path/'receipt.json';sp.write_bytes(m.raw(s))
    op=tmp_path/'original.json';op.write_bytes(m.raw(old));os=tmp_path/'oldstrict.json';os.write_bytes(m.raw(old_s))
    # Only RAM authority hashes substituted, no production artifacts used.
    monkeypatch.setattr(m,'OLD_REPORT_SHA',m.sha(op));old_s['report_sha256']=m.OLD_REPORT_SHA;os.write_bytes(m.raw(old_s))
    monkeypatch.setattr(m,'OLD_STRICT_SHA',m.sha(os))
    p['planner_authority'].update(report_sha256=m.OLD_REPORT_SHA,receipt_sha256=m.OLD_STRICT_SHA)
    s['planner_authority']=deepcopy(p['planner_authority']);r['execution_binding']['planner_authority']=deepcopy(p['planner_authority'])
    pp.write_bytes(m.raw(p));r['execution_binding']['prepare_sha256']=m.sha(pp)
    rp.write_bytes(m.raw(r));s['physical_report_sha256']=m.sha(rp)
    s['bindings'].update({str(pp):m.sha(pp),str(rp):m.sha(rp)});sp.write_bytes(m.raw(s))
    out=tmp_path/(m.NAME+'_fixture.json')
    cli=['analyze','--report',str(rp),'--report-sha',m.sha(rp),'--receipt',str(sp),'--receipt-sha',m.sha(sp),
        '--original-report',str(op),'--original-strict',str(os),'--output',str(out)]
    m.main(cli)
    m.main(['verify','--summary',str(out),'--summary-sha',m.sha(out)])
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['verified']
    with pytest.raises(FileExistsError):m.main(cli)
    journal.write_bytes(journal.read_bytes()+b'{}\n')
    with pytest.raises(ValueError):m.main(cli)


def test_bool_nan_raw_and_signed_zero():
    for v in (True,float('nan'),.1,2.):
        with pytest.raises(ValueError):m.action_bits([[v,0.]],1)
    assert m.action_bits([[-0.,0.]],1)!=m.action_bits([[0.,0.]],1)
