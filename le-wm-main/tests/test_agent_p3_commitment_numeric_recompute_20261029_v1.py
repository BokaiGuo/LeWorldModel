"""Standalone arithmetic/log fixtures: no producer consumer, torch or physics imports."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_commitment_numeric_recompute_20261029_v1 as m


def fixture(tmp_path):
    output=tmp_path/'synthetic';output.mkdir();events=[];serial=0
    counts={r:dict(attempted=0,completed=0) for r in m.PHYSICAL}
    def emit(kind,**fields):
        events.append(dict(index=len(events),kind=kind,previous_sha256=events[-1]['event_sha256'] if events else None,**fields))
        events[-1]['event_sha256']=m.digest(events[-1],compact=False)
        return events[-1]
    def operation(eid,resource,region,body=lambda:None):
        nonlocal serial
        i=serial;serial+=1;kwargs=dict(episode_id=eid,resource=resource,region=region,operation_id=i,units=1)
        emit('operation_attempted',**kwargs);counts[resource]['attempted']+=1;body()
        emit('operation_completed',**kwargs);counts[resource]['completed']+=1
    def write(path,value):path.write_bytes(m.canonical(value)+b'\n')
    contract={'budget_match':'notmatched','fixture_not_execution':True}
    emit('run_started',contract=contract);rows=[];nodes=[];per=[]
    for slot,mode in m.KEYS:
        eid=f's{slot}_{mode}';mode_id=m.MODES.index(mode)
        operation(eid,'constructor','setup')
        for _ in range(2):
            def reset():
                for _ in range(2):operation(eid,'set_state','setup')
                operation(eid,'goal_setter','setup');operation(eid,'goal_render','setup')
                for _ in range(2):operation(eid,'physics_step','setup')
            operation(eid,'reset','setup',reset)
        operation(eid,'explicit_render','setup')
        for _ in range(10):
            operation(eid,'public_step','warmup',lambda:[operation(eid,'physics_step','warmup') for _ in range(10)])
            operation(eid,'explicit_render','warmup')
        operation(eid,'goal_bind','goal_bind',lambda:operation(eid,'goal_setter','goal_bind'))
        operation(eid,'explicit_render','goal_bind')
        goal=[0.]*7;last=None;executed=[]
        for step in range(21):
            state=[100.+slot+(max(0,step-5)*mode_id),20.,40.,50.,.3,1.,2.]
            receipt=dict(phase='plan',step=step,state=state,goal=goal,snapshot={'x':state},goal_rgb_sha256='e'*64,
                goal_pose=[20.,30.,.1],pixel_sha256=m.digest([slot,step] if step<=5 else [slot,mode,step]),
                observed_event_index=len(events),path='synthetic_only')
            if step:
                command=np.asarray([.1,0.],np.float32).tolist();executed.append(command)
                emit('authorized_dispatch',episode_id=eid,step=step,command=command)
                operation(eid,'public_step','prefix' if step<=5 else 'suffix',
                    lambda:[operation(eid,'physics_step','prefix' if step<=5 else 'suffix') for _ in range(10)])
                operation(eid,'explicit_render','prefix' if step<=5 else 'suffix')
                receipt['observed_event_index']=len(events)
            emit('fresh_observation',episode_id=eid,receipt=receipt);last=receipt
            if step:emit('control_return',episode_id=eid,step=step,command=command,
                reward=-float(np.linalg.norm(np.asarray(state))),terminated=False,truncated=False)
        original=m.metrics(goal,last['state'])['original']
        row=dict(episode_id=eid,source_slot=slot,mode=mode,status='completed20',executed_raw=executed,
            control_returned=20,warmup_returned=10,full_horizon=True,final_state=last['state'],last_observation=last,
            final_metrics=original,native_success=False,terminated=False,truncated=False)
        rows.append(row);operation(eid,'close','closing')
        arm={k:dict.fromkeys(m.AXES,0) for k in ('attempted','completed','signal_completed','program_completed')}
        for boundary in (5,10,15):
            # Native-shaped paid-call log; intentionally NOT a model or full compiler fixture.
            ns=output/(eid+f'_b{boundary:02}');ns.mkdir();ops=[];native=[];previous='0'*64
            def nevent(payload):
                nonlocal previous
                e=dict(index=len(native),previous=previous,payload=payload);e['sha256']=m.digest(e)
                previous=e['sha256'];native.append(e)
            for resource,shape,directions in [('predict',[1,3,192],None),('grad',None,None),('vjp',None,192)]:
                d=dict.fromkeys(m.AXES,0)
                if resource=='predict':d.update(predict_API=1,predict_samples=1,predict_context_tokens=3)
                elif resource=='grad':d['grad_API']=1
                else:d['VJP_directions']=192
                op=dict(index=len(ops),resource=resource,shape=shape,directions=directions,delta=d,
                    scope={'slot':slot,'horizon':1},status='attempted',started_ns=10,ended_ns=None,error=None)
                nevent(dict(kind='resource_attempt',entry=deepcopy(op)))
                op.update(status='returned',ended_ns=20);nevent(dict(kind='resource_return',entry=deepcopy(op)));ops.append(op)
                for k in m.AXES:
                    for f in ('attempted','completed','signal_completed'):arm[f][k]+=d[k]
            path=ns/'events.jsonl';path.write_bytes(b''.join(m.canonical(e)+b'\n' for e in native))
            rr=dict(events=ops,caps={k:1000 for k in m.AXES},
                attempted={k:sum(op['delta'][k] for op in ops) for k in m.AXES},
                completed={k:sum(op['delta'][k] for op in ops) for k in m.AXES})
            nodes.append(dict(episode_id=eid,boundary=boundary,source_slot=slot,mode=mode,status='sealed',
                native_journal_sha256=m.sha(path),native_journal_tail=previous,resource_receipt=rr))
        per.append(dict(source_slot=slot,mode=mode,receipts=3,**arm))
    emit('run_finished',status='known_fullcommit_recorded',counts=counts,nodes_sha256=m.digest(nodes))
    (output/'journal.jsonl').write_bytes(b''.join(m.canonical(e,compact=False)+b'\n' for e in events))
    prepared=tmp_path/'prepare.json';write(prepared,dict(schema=m.PRODUCER+'.prepare',contract=contract))
    r=dict(schema=m.PRODUCER+'.report',contract=contract,prepared=str(prepared),prepare_sha256=m.sha(prepared),
        rows=rows,nodes=nodes,per_episode_neural=per,status='known_fullcommit_recorded',errors=[],pending={},
        fixed_denominator=6,node_denominator=18,counts=counts,journal_sha256=m.sha(output/'journal.jsonl'),
        journal_chain_tail=events[-1]['event_sha256'],budget_match='notmatched',equal_FLOPs=False,equal_wall=False,
        **{k:False for k in ('P1_admission','P3_admission','freshness','scientific_admission')})
    write(output/'report.json',r)
    strict=tmp_path/'strict.json';write(strict,dict(schema=m.PRODUCER+'.consume',complete=True,counts=counts,
        fixed_denominator=6,node_denominator=18,per_episode_neural=per,native_metrics=[x['final_metrics'] for x in rows],
        **{k:False for k in ('P1_admission','P3_admission','freshness','scientific_admission')}))
    return output,prepared,strict,r,events


def analyze(b):
    out,p,s,_,_=b
    return m.analyze(out,m.sha(out/'report.json'),p,m.sha(p),s,m.sha(s))


def test_full_six_numeric_recompute_and_all_six_pairs(tmp_path):
    result=analyze(fixture(tmp_path))
    assert result['complete'] and len(result['rows'])==6 and len(result['pairs'])==6
    assert result['paid']['physical']['total']['public_step']==dict(attempted=180,completed=180)
    assert result['paid']['physical']['total']['physics_step']==dict(attempted=1824,completed=1824)
    assert len(result['paid']['nodes'])==18
    assert all(x['paired_prefix_available'] and x['budget_match']=='notmatched' for x in result['pairs'])
    assert all(r['FLOPs'] is None and r['completed']['VJP_directions']==576 for r in result['paid']['native'].values())
    assert not result['independent_confirmation'] and not result['P3_admission']


@pytest.mark.parametrize('angle',[0.,.2,np.pi,2*np.pi-.01,7.])
def test_native7_original_angle_not_wrapped_replacement(angle):
    state=[0.,0.,0.,0.,angle,3.,4.];r=m.metrics([0.]*7,state)
    assert r['original']['native_distance']==float(np.linalg.norm(state))
    assert r['velocity_error']==5. and r['joint_xy']==0.
    assert r['original']['native_angle_term']==min(angle,2*np.pi-angle)
    assert r['original']['native_success']==bool(min(angle,2*np.pi-angle)<np.pi/9)


@pytest.mark.parametrize('damage',['metric','goal','counter','denominator','pair','matched','source_sha','delta','native_payload'])
def test_recomputed_sha_transport_does_not_hide_wrong_numbers(tmp_path,damage):
    b=fixture(tmp_path);out,p,s,r,events=b
    if damage=='metric':r['rows'][0]['final_metrics']['native_distance']+=1
    elif damage=='goal':next(e for e in events if e['kind']=='fresh_observation')['receipt']['goal'][0]=1
    elif damage=='counter':r['counts']['physics_step']['completed']-=1
    elif damage=='denominator':r['rows'].pop()
    elif damage=='pair':next(e for e in events if e['kind']=='fresh_observation' and e['episode_id']=='s0_fixedH3'
        and e['receipt']['step']==5)['receipt']['snapshot']['x'][0]+=1
    elif damage=='matched':r['budget_match']='matched'
    elif damage=='source_sha':
        with pytest.raises(ValueError):m.analyze(out,'0'*64,p,m.sha(p),s,m.sha(s))
        return
    else:
        path=out/'s0_fixedH1_b05/events.jsonl';native=[json.loads(x) for x in path.read_bytes().splitlines()]
        if damage=='delta':native[0]['payload']['entry']['delta']['predict_API']=9
        else:native[0]['payload']['entry']['shape']=[99,3,192]
        previous='0'*64
        for e in native:
            e['previous']=previous;e['sha256']=m.digest({k:v for k,v in e.items() if k!='sha256'});previous=e['sha256']
        path.write_bytes(b''.join(m.canonical(e)+b'\n' for e in native))
        r['nodes'][0].update(native_journal_sha256=m.sha(path),native_journal_tail=previous)
        events[-1]['nodes_sha256']=m.digest(r['nodes'])
    prev=None
    for e in events:
        e['previous_sha256']=prev;e['event_sha256']=m.digest({k:v for k,v in e.items() if k!='event_sha256'},compact=False)
        prev=e['event_sha256']
    (out/'journal.jsonl').write_bytes(b''.join(m.canonical(e,compact=False)+b'\n' for e in events))
    r.update(journal_sha256=m.sha(out/'journal.jsonl'),journal_chain_tail=prev)
    (out/'report.json').write_bytes(m.canonical(r))
    with pytest.raises(ValueError):analyze(b)


def test_real_journal_codecs_double_lf_and_bad_json_rejected(tmp_path):
    b=fixture(tmp_path);out=b[0]
    for path,native in ((out/'journal.jsonl',False),(out/'s0_fixedH1_b05/events.jsonl',True)):
        original=path.read_bytes();events,_=m.journal(path,native=native);assert events
        path.write_bytes(original.replace(b'\n',b'\n\n',1))
        with pytest.raises(ValueError):m.journal(path,native=native)
        path.write_bytes(original+b'bad\n')
        with pytest.raises(ValueError):m.journal(path,native=native)
        path.write_bytes(original)


def test_no_producer_imports():
    import ast
    tree=ast.parse(Path(m.__file__).read_text())
    modules=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert all(x in ('__future__','collections','pathlib') for x in modules)


def test_missing_metric_pair_preserved_not_filtered():
    rows=[dict(episode_id=f's{s}_{mode}',source_slot=s,mode=mode,actual_control_returned=0,
        prefix5=None,goal=None,metrics=None,native_success=None) for s,mode in m.KEYS]
    pairs=m.comparisons(rows)
    assert len(pairs)==6 and all(x['delta_right_minus_left'] is None for x in pairs)


@pytest.mark.parametrize('resource,shape,directions',[('weird',None,None),('grad',[1],None),('predict',[1,2,192],None),('vjp',None,True)])
def test_charge_unknown_or_wrong_units_rejected(resource,shape,directions):
    with pytest.raises(ValueError):m.delta(dict(resource=resource,shape=shape,directions=directions,delta={}))
