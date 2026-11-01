"""Small synthetic saved arrays/events only; no original production consume."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_effective_schedule_summary_20261029_v1 as m


def literal(value):return float(np.float32(value))


def fixture(*,short=None):
    """Deliberately synthetic reduced event data, NOT a producer certification.

    Resource deltas are aggregated fixture operations, not model invocations.
    Production analyze MUST call the real owner.consume before any arithmetic.
    """
    goal=[1.]+[0.]*191;root=np.zeros((3,192),np.float32).tolist()
    base=dict(goal=goal,projection=np.eye(192).tolist(),root_context=dict(latent_history=root))
    p=dict(tasks=[deepcopy(base),deepcopy(base)],sources={})
    rows=[];nodes=[];events=[];arrays={};serial=0
    native_axes=m.owner.contract()['physical_caps']|m.owner.contract()['visual_caps']
    for index,(s,arm) in enumerate((s,a) for s in (0,1) for a in m.owner.ARMS):
        eid=f's{s}_{arm}';schedule,bank=arm.split('_',1)
        n=14 if short==(s,arm) else 20
        row=dict(episode_id=eid,source_slot=s,mode=arm,status='native_stopped_control' if n<20 else 'completed20',
            warmup_returned=10,control_returned=n,initial_exact=True,terminated=False,truncated=n<20,
            full_horizon=n==20,encodings=[],executed_raw=[])
        rows.append(row);latents={}
        for b in (5,10,15,20):
            if b>=n and n<20:continue  # stop at boundary itself is not encoded by real episode
            z=np.zeros(192,np.float32);z[0]=np.float32(b/100+(index%4)*.01 if b>5 else .05)
            key=eid+str(b);arrays[key]=z;latents[b]=z.tolist()
            row['encodings'].append(dict(step=b,latent=dict(key=key)))
        for b in (5,10,15):
            node=dict(episode_id=eid,source_slot=s,mode=arm,boundary=b,status='not_reached',
                request=None,receipt=None,resource_receipt=None)
            nodes.append(node)
            if b>=n and n<20:continue
            ctx=dict(latent_history=[root[0],root[1],latents[b]])
            node['request']=dict(task=dict(root_context=ctx))
            delta=m.owner.costs(arm,b)
            rr=dict(events=[dict(status='returned',delta=delta,resource='SYNTHETIC_AGGREGATE',started_ns=0,ended_ns=1)],
                attempted=deepcopy(delta),completed=deepcopy(delta),inclusive_operation_wall_ns={'SYNTHETIC_AGGREGATE':1})
            node['resource_receipt']=rr
            status='observed_commitment' if schedule=='commit15' and b>5 else 'sealed'
            node['status']=status
            result=dict(status=status,decision=None,bank=None,candidates=[],winner=None)
            node['receipt']=result
            if status=='sealed':
                h=3 if schedule=='commit15' else 1;actions=[];candidates=[]
                for i,f in enumerate(m.owner.BANKS[bank]):
                    raw=np.full((h*5,2),np.float32(f*.1)).tolist();actions.append(raw)
                    z=np.zeros(192,np.float32);z[0]=np.float32(f*.8)
                    terminal=dict(latent_history=[root[0],root[1],z.tolist()])
                    candidates.append(dict(index=i,actions=raw,whole=[terminal]*h,cost=m.latent_q(goal,z.tolist())))
                result.update(decision=dict(horizon=h,candidates=4,bank=bank,schedule=schedule),
                    bank=dict(actions=actions,fractions=list(m.owner.BANKS[bank])),candidates=candidates,winner=dict(index=3))
        executed=[np.full(2,np.float32(.05)).tolist() for _ in range(5)]
        for b in (5,10,15):
            origin=next(node for node in nodes if node['episode_id']==eid and node['boundary']==(5 if schedule=='commit15' else b))
            if origin['receipt'] is None:break
            selected=origin['receipt']['candidates'][3]['actions']
            offset=b-5 if schedule=='commit15' else 0
            executed.extend(deepcopy(selected[offset:offset+5]))
        row['executed_raw']=executed[:n]
        for b in range(n+1):
            state=[float(b)+index*.1,float(b)/2,2.*b,3.*b, .1*b,float(b)/10,-float(b)/20]
            ob=dict(phase='plan',step=b,state=state,goal=[100.,100.,100.,100.,0.,0.,0.],
                pixel_sha256=m.digest([s,b,index%4 if b>5 else 0,'pixels']),
                file_sha256=m.digest([s,b,index%4 if b>5 else 0,'png']))
            events.append(dict(kind='fresh_observation',episode_id=eid,receipt=ob))
            if b:
                events.append(dict(kind='authorized_dispatch',episode_id=eid,step=b,command=row['executed_raw'][b-1]))
                events.append(dict(kind='control_return',episode_id=eid,step=b,command=row['executed_raw'][b-1]))
        want=dict(constructor=1,reset=2,set_state=4,goal_render=2,goal_setter=3,goal_bind=1,
            public_step=10+n,physics_step=4+10*(10+n),explicit_render=12+n,close=1,
            encoder=len(row['encodings']),projector=len(row['encodings']))
        assert set(want)==set(native_axes)
        for k,u in want.items():
            for kind in ('operation_attempted','operation_completed'):
                events.append(dict(kind=kind,episode_id=eid,resource=k,operation_id=serial,units=u,region='SYNTHETIC'))
            serial+=1
    counts={k:dict(attempted=0,completed=0) for k in native_axes}
    for e in events:
        if e['kind'] in ('operation_attempted','operation_completed'):
            counts[e['resource']]['attempted' if e['kind']=='operation_attempted' else 'completed']+=e['units']
    report=dict(rows=rows,nodes=nodes,counts=counts,wall_seconds=123.,per_episode_neural=[])
    for row in rows:
        local=[n['resource_receipt'] for n in nodes if n['episode_id']==row['episode_id'] and n['resource_receipt'] is not None]
        report['per_episode_neural'].append(dict(source_slot=row['source_slot'],mode=row['mode'],
            attempted={k:sum(r['attempted'][k] for r in local) for k in m.owner.core.AXES},
            completed={k:sum(r['completed'][k] for r in local) for k in m.owner.core.AXES},
            inclusive_operation_wall_ns={'SYNTHETIC_AGGREGATE':len(local)}))
    return p,report,events,arrays


def summary(**kwargs):
    p,r,e,a=fixture(**kwargs)
    return m.summarize(p,r,e,lambda entry:a[entry['key']])


def test_complete_fixed_rows_nodes_pairs_and_all_axes():
    s=summary()
    assert len(s['rows'])==8 and sum(len(r['nodes']) for r in s['rows'])==24 and len(s['pairs'])==8
    assert [p['contrast'] for p in s['pairs']]==['same_bank_schedule']*2+['same_schedule_bank']*2+['same_bank_schedule']*2+['same_schedule_bank']*2
    assert all(len(p['steps'])==21 and len(p['checkpoints'])==5 for p in s['pairs'])
    for row in s['rows']:
        assert row['counts']['neural']['predict_API']==dict(attempted=51,completed=51)
        assert row['counts']['neural']['jacobian_API']==dict(attempted=6,completed=6)
        assert row['counts']['neural']['VJP_directions']==dict(attempted=1152,completed=1152)
        assert set(row['counts']['neural'])==set(m.owner.core.AXES)
    assert s['independent_samples'] is None and s['known_roots']==2
    assert s['P1_admission'] is s['P3_admission'] is s['C3'] is s['equal_FLOPs'] is False


def test_earlystop_same_time_only_and_no_terminal_padding():
    s=summary(short=(1,'feedback5_midpoints'))
    row=next(r for r in s['rows'] if r['episode_id']=='s1_feedback5_midpoints')
    assert row['missing_steps']==list(range(15,21)) and row['eventtime_terminal']['step']==14
    assert row['steps'][15] is row['steps'][20] is None
    assert row['nodes'][2]['status']=='not_reached' and row['nodes'][2]['bank'] is None
    pair=next(p for p in s['pairs'] if p['right']==row['episode_id'] and p['contrast']=='same_bank_schedule')
    assert pair['common_steps']==list(range(15))
    assert pair['eventtime_terminal']==dict(left_step=20,right_step=14,same_time=False,right_minus_left=None)
    assert pair['steps'][15]['right_minus_left'] is None and pair['steps'][15]['actual_Q_delta'] is None
    assert pair['support'][2]['right'] is None
    origin=next(r for r in s['rows'] if r['episode_id']=='s1_commit15_midpoints')['nodes'][0]
    assert origin['actual_Q_at_predicted_endpoint'] is not None
    h1=row['nodes'][1]
    assert h1['predicted_endpoint_step']==15 and h1['actual_Q_at_predicted_endpoint'] is None


def test_support_origins_and_no_crossbank_causal_label():
    s=summary();p=s['pairs'][0]
    support=p['support'][1]
    assert support['left']['origin_boundary']==5 and support['left']['fresh_at_boundary'] is False
    assert support['right']['origin_boundary']==10 and support['right']['fresh_at_boundary'] is True
    assert support['same_compile_horizon'] is False and support['compile_context_equal'] is False
    assert len(support['packet_raw_support_overlap'])==4
    for pair in s['pairs']:
        if pair['contrast']=='same_schedule_bank':
            assert 'NOT pure scheduling causality' in pair['interpretation']
            assert pair['support'][0]['packet_raw_support_overlap']==[]


def test_Q_is_independent_full192_and_endpoint_semantics():
    p,r,e,a=fixture();s=m.summarize(p,r,e,lambda entry:a[entry['key']])
    row=s['rows'][0]
    assert row['steps'][0]['Q_source'].startswith('sealed_original_root_encoding')
    assert row['steps'][1]['actual_Q'] is None
    assert row['nodes'][0]['predicted_endpoint_step']==20
    assert row['nodes'][0]['actual_Q_at_predicted_endpoint']==row['steps'][20]['actual_Q']
    z=deepcopy(p['tasks'][0]['goal']);z[-1]=2.
    assert m.latent_q(p['tasks'][0]['goal'],z)==4/192


def test_native7_keeps_velocity_and_source_native_angle_not_redefined():
    g=[0.]*7;state=[3.,4.,0.,0.,7.,6.,8.]
    v=m.physical_metric(g,state)
    assert v['joint_xy']==5 and v['object_xy']==0 and v['velocity_error']==10
    assert v['native7_distance']==float(np.sqrt(174))
    assert v['native_angle_term']<0 and v['angle_wrapped']>0


@pytest.mark.parametrize('bad',([True]*7,[float('nan')]*7,[0.]*6))
def test_bad_metric_vectors_reject(bad):
    with pytest.raises(ValueError):m.physical_metric([0.]*7,bad)


@pytest.mark.parametrize('damage',('rows','nodes','raw','Q','delta','paid','counter','projection'))
def test_numeric_fixture_mutation_rejection(damage):
    p,r,e,a=fixture()
    if damage=='rows':r['rows'].pop()
    elif damage=='nodes':r['nodes'].pop()
    elif damage=='raw':r['rows'][0]['executed_raw'][0][0]=.123456789
    elif damage=='Q':r['nodes'][0]['receipt']['candidates'][0]['cost']+=1
    elif damage=='delta':r['nodes'][0]['resource_receipt']['events'][0]['delta']['predict_API']+=1
    elif damage=='paid':r['per_episode_neural'][0]['completed']['predict_API']+=1
    elif damage=='counter':r['counts']['physics_step']['completed']+=1
    else:p['tasks'][0]['projection'][0][0]=2.
    with pytest.raises(ValueError):m.summarize(p,r,e,lambda entry:a[entry['key']])


def authority_fixture(tmp_path,monkeypatch,*,wrong_receipt=False):
    p,r,e,a=fixture();output=tmp_path/'input';output.mkdir()
    prepared=tmp_path/'prepare.json';strict=tmp_path/'strict.json'
    dep=tmp_path/'binding';dep.write_bytes(b'sealed immutable input')
    p['sources']={str(dep):m.sha(dep)}
    receipt=dict(complete=True,fixed_denominator=8,node_denominator=24,fixture=True)
    (output/'report.json').write_bytes(m.canonical(r));prepared.write_bytes(m.canonical(p))
    strict.write_bytes(m.canonical(dict(receipt,fixture=False) if wrong_receipt else receipt))
    own=tmp_path/'own_fixture';own.write_bytes(b'fixture source')
    monkeypatch.setattr(m,'own_files',lambda:[own]);calls=[]
    def consume(*args):calls.append(args);return deepcopy(receipt)
    monkeypatch.setattr(m.owner,'consume',consume)
    monkeypatch.setattr(m.owner,'load_events',lambda *args:e)
    monkeypatch.setattr(m.owner,'load_array',lambda out,entry:a[entry['key']])
    args=(output,m.sha(output/'report.json'),prepared,m.sha(prepared),strict,m.sha(strict))
    return args,calls,dep


def test_analyze_FULL_own_strict_rebuild_called_and_verify_deterministic(tmp_path,monkeypatch):
    args,calls,dep=authority_fixture(tmp_path,monkeypatch)
    first=m.analyze(*args);assert calls==[(args[0],args[1],args[2],args[3])]
    assert first['authority']['full_own_strict_rebuilt_here'] and first['bindings'][str(dep)]==m.sha(dep)
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    path=tmp_path/'results'/(m.NAME+'_RAM.json')
    receipt=m.write(path,first)
    assert m.verify(path,receipt['sha256'])['verified'] and len(calls)==2
    with pytest.raises(ValueError):m.write(path,first)
    dep.write_bytes(b'drift')
    with pytest.raises(ValueError):m.analyze(*args)
    assert len(calls)==2


def test_status_only_or_nonmatching_strict_rejected(tmp_path,monkeypatch):
    args,calls,_=authority_fixture(tmp_path,monkeypatch,wrong_receipt=True)
    with pytest.raises(ValueError):m.analyze(*args)
    assert len(calls)==1


@pytest.mark.parametrize('which',(1,3,5))
def test_wrong_external_SHA_rejected_before_consume(tmp_path,monkeypatch,which):
    args,calls,_=authority_fixture(tmp_path,monkeypatch);args=list(args);args[which]='0'*64
    with pytest.raises(ValueError):m.analyze(*args)
    assert not calls


def test_frozen_owner_drift_rejected_before_consume(tmp_path,monkeypatch):
    args,calls,_=authority_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'PINS',('0'*64,)*3)
    with pytest.raises(ValueError):m.analyze(*args)
    assert not calls


def test_import_has_no_torch_or_physics():
    script=f'import sys;import experiments.spectral_reachability.{m.NAME};assert "torch" not in sys.modules;assert "pymunk" not in sys.modules'
    subprocess.run([sys.executable,'-c',script],cwd=m.ROOT,check=True)
