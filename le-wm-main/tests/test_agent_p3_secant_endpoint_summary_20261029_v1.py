"""Synthetic receipt fixtures only; never production saved analysis."""
import ast
import copy
import json
import math
from pathlib import Path
import struct

import pytest
from experiments.spectral_reachability import agent_p3_secant_endpoint_summary_20261029_v1 as m


def checkpoint(step,state):
    return dict(step=step,state=state,snapshot=dict(state=state,state_float64_bits=struct.pack('<7d',*state).hex()),
        png=f'fixture_{step}.png',file_sha256='pnghash',pixel_sha256=f'pixels{step}')


def fixture_receipts():
    def row(i,root,arm,missing=False):
        goal=[0.]*7
        plan=dict(index=i,root_slot=root,arm=arm,condition='off',task_id=f'task{root}',local_task_id=f'local{root}',
            source_slot=root,native_goal=goal,native_goal_pose=[0.,0.,0.],native_goal_pixel_sha256='goalpixels',
            native_root_sha256=f'root{root}',prefix_commands=[[0.,0.]]*5,base_task={'root':root},model_task={'root':root},
            provenance=dict(actual_cp5_absolute_steps=[5,10,15],actual_cp5_control_steps=[-5,0,5]),
            raw_steps=0 if missing else 10,selection_uses_physics=False,commands=None if missing else [[0.,0.]]*10)
        if arm in ('Z','P'):plan['control_role']=arm
        status='no_witness' if missing else 'horizon_completed'
        # All native-success values here are independently obvious, not fixture calls to metrics.
        state=[3.,4.,0.,0.,.25,1.,2.] if arm in ('H1','H3') else [1.,2.,0.,0.,.125,0.,0.]
        cps=[] if missing else [checkpoint(0,[0.]*7),checkpoint(5,[1.]*7),checkpoint(10,state)]
        return dict(index=i,root_slot=root,arm=arm,condition='off',task_id=f'task{root}',plan=plan,
            status=status,checkpoints=cps,responses=[] if missing else [dict(step=10,state=state,native_success=True)],
            goal=goal,root_sha256=f'root{root}',prefix_reference_exact=not missing,
            own_attempted=0 if missing else 10,own_returned=0 if missing else 10,PNG=0 if missing else 66,
            final_state=None if missing else state,final_metrics=None if missing else {'fixture':True},native_success=None if missing else True)
    crows=[row(i,root,arm,i!=0) for i,(root,arm) in enumerate(m.ORDER_C)]
    zrows=[row(i,root,arm) for i,(root,arm) in enumerate(m.ORDER_ZP)]
    def envelope(name,status,rows):
        return dict(schema=name+'.consume',status=status,rows=rows,errors=[],fixed_denominator=4,known_roots=2,
            fresh_roots=0,inherited=0,completion=dict(status='published_within_cap',report_sha256='report'),
            physical_report_sha256='report',counts={'public':dict(attempted=10,completed=10)},saved_PNG_verified=66,
            **{k:False for k in m.FALSE_FLAGS})
    c=envelope(m.CNAME,'secant_commit5_native_recorded_NOT_admitted',crows)
    z=envelope(m.ZNAME,'all4_ZP_first5_native_recorded_NOT_admitted',zrows)
    refs={'C':dict(path='/fixture/C.json',sha256='c'*64),'ZP':dict(path='/fixture/ZP.json',sha256='d'*64)}
    z['planner_authority']=dict(authorities={'C_receipt':refs['C']},logical_C_slots=[dict(root_slot=r['root_slot'],arm=r['arm'],status=r['status'],
        endpoint=None if r['status']=='no_witness' else r['checkpoints'][-1]) for r in crows])
    z['endpoints']=[dict(index=r['index'],root_slot=r['root_slot'],role=r['arm'],control_step=10,absolute_step=20,
        available=True,checkpoint=r['checkpoints'][-1],missing_reason=None) for r in zrows]
    return c,z,refs


def test_fsum_native_all_axes():
    state=[3.,4.,0.,12.,.5,8.,15.];goal=[0.]*7
    out=m.metrics(state,goal)
    assert out['native7_squared']==25+144+.25+64+225
    assert out['joint_xy_error']==13 and out['agent_xy_error']==5 and out['block_xy_error']==12
    assert out['agent_velocity_error']==17
    assert out['native7_reward']==-math.sqrt(458.25)
    assert out['group_roundoff_residual']==0
    assert out['signed_residual7']==state and out['axis_order'][-2:]==['agent_vx','agent_vy']


@pytest.mark.parametrize('x,angle,expected',[(19.,0.,True),(20.,0.,False),(0.,math.pi/9,False),(0.,7.,True)])
def test_original_native_strict_rule(x,angle,expected):
    out=m.metrics([x,0.,0.,0.,angle,999.,-999.],[0.]*7)
    assert out['native_success'] is expected
    if angle==7.:
        assert out['negative_native_angle'] and not out['wrapped_success_diagnostic']


@pytest.mark.parametrize('bad',[[0.]*6,[True]+[0.]*6,[float('nan')]+[0.]*6,[float('inf')]+[0.]*6,['0']+[0.]*6])
def test_bad_native_vectors(bad):
    with pytest.raises(ValueError):m.metrics(bad,[0.]*7)


@pytest.mark.parametrize('a,b,larger,cw,zw,tie',[(2.,1.,True,True,False,False),(1.,2.,True,False,True,False),
    (-2.,-1.,False,True,False,False),(0.,-0.,True,False,False,True),(5e-324,-5e-324,True,True,False,False)])
def test_strict_directions(a,b,larger,cw,zw,tie):
    out=m.compare(a,b,larger)
    assert (out['C_worse'],out['control_worse'],out['tie'])==(cw,zw,tie)


def test_null_comparison_not_tie():
    x=m.compare(None,1.)
    assert x['status']=='null' and x['tie'] is None and x['C_worse'] is None


def test_fixed8_all_null_slots_retained():
    c,z,refs=fixture_receipts();out=m.summarize(c,z,refs)
    assert len(out['comparisons'])==8 and len(out['C_slots'])==len(out['ZP_root_references'])==4
    assert out['denominator']==dict(C_logical_slots=4,C_no_witness=3,ZP_references=4,comparisons=8)
    assert [p['available'] for p in out['comparisons']]==[True,True]+[False]*6
    for p in out['comparisons'][:2]:
        assert p['errors']['joint_xy_error']['C_worse']
        assert p['reward']['C_worse'] and not p['reward']['control_worse']
    assert out['comparison_counts']['joint_xy_error']==dict(C_worse=2,control_worse=0,tie=0,null=6)
    assert out['comparison_counts']['reward']==dict(C_worse=2,control_worse=0,tie=0,null=6)
    assert out['comparison_counts']['native_success']==dict(C_worse=0,control_worse=0,tie=2,null=6)
    assert out['baseline'] is None and out['cp5_is_baseline'] is False and out['actual_time']==dict(control=10,absolute=20)


@pytest.mark.parametrize('what',['schema','errors','flag','root','arm','time','goal','prefix','authority','logical','endpoint','statebits','cp5substitute','borrowmissing','success'])
def test_semantic_mutations(what):
    c,z,refs=fixture_receipts()
    if what=='schema':z['schema']='other'
    elif what=='errors':z['errors']=['partial']
    elif what=='flag':c['P3_admission']=True
    elif what=='root':z['rows'][0]['root_slot']=1
    elif what=='arm':c['rows'][0]['arm']='H3'
    elif what=='time':z['endpoints'][0]['absolute_step']=15
    elif what=='goal':c['rows'][1]['plan']['native_goal']=[1.]*7
    elif what=='prefix':z['rows'][0]['checkpoints'][1]['pixel_sha256']='other'
    elif what=='authority':z['planner_authority']['authorities']['C_receipt']={'path':'other','sha256':'other'}
    elif what=='logical':z['planner_authority']['logical_C_slots'][1]['endpoint']=c['rows'][0]['checkpoints'][-1]
    elif what=='endpoint':z['endpoints'][0]['checkpoint']={'step':5}
    elif what=='statebits':z['rows'][0]['checkpoints'][-1]['snapshot']['state_float64_bits']='00'
    elif what=='cp5substitute':c['rows'][0]['checkpoints'][-1]['step']=5
    elif what=='borrowmissing':c['rows'][1]['final_state']=[0.]*7
    else:c['rows'][0]['native_success']=False
    with pytest.raises(ValueError):m.summarize(c,z,refs)


def test_early_stop_not_same_time_endpoint():
    c,z,refs=fixture_receipts();r=z['rows'][0]
    r.update(status='native_stopped',own_returned=8,final_state=[9.]*7)
    r['checkpoints']=r['checkpoints'][:2];r['responses']=[]
    z['endpoints'][0].update(available=False,checkpoint=None,missing_reason='native_stopped')
    out=m.summarize(c,z,refs)
    assert not out['comparisons'][0]['available']
    assert out['comparisons'][0]['errors']['native7_distance']['C_worse'] is None


def test_native_stop_exactly10_still_endpoint():
    c,z,refs=fixture_receipts();z['rows'][0]['status']='native_stopped'
    assert m.summarize(c,z,refs)['comparisons'][0]['available']


def test_position_velocity_not_combined_claim():
    c,z,refs=fixture_receipts()
    # C position closer, velocity much larger: preserve opposite directions.
    s=[0.,0.,0.,0.,0.,100.,100.];row=c['rows'][0]
    row['checkpoints'][-1]=checkpoint(10,s);row['responses'][0]['state']=s;row['final_state']=s
    z['planner_authority']['logical_C_slots'][0]['endpoint']=row['checkpoints'][-1]
    out=m.summarize(c,z,refs)['comparisons'][0]
    assert out['errors']['joint_xy_error']['control_worse']
    assert out['errors']['native7_distance']['C_worse']


def test_success_worse_direction_independent_of_reward():
    c,z,refs=fixture_receipts();s=[20.,0.,0.,0.,0.,0.,0.];row=c['rows'][0]
    row['checkpoints'][-1]=checkpoint(10,s);row['responses'][0].update(state=s,native_success=False)
    row['final_state']=s;row['native_success']=False
    z['planner_authority']['logical_C_slots'][0]['endpoint']=row['checkpoints'][-1]
    out=m.summarize(c,z,refs)
    assert out['comparisons'][0]['native_success']['C_worse']
    assert not out['comparisons'][0]['native_success']['control_worse']
    assert out['comparison_counts']['native_success']==dict(C_worse=2,control_worse=0,tie=0,null=6)


def fixture_files(tmp_path):
    c,z,refs=fixture_receipts();folder=tmp_path/'input';folder.mkdir()
    cp=folder/'C.json';cp.write_bytes(m.encode(c));refs['C']=dict(path=str(cp),sha256=m.sha(cp))
    z['planner_authority']['authorities']['C_receipt']=refs['C']
    zp=folder/'ZP.json';zp.write_bytes(m.encode(z));refs['ZP']=dict(path=str(zp),sha256=m.sha(zp))
    return refs


def test_fixture_build_write_verify(tmp_path,monkeypatch):
    refs=fixture_files(tmp_path);monkeypatch.setattr(m,'actual_refs',lambda:refs)
    monkeypatch.setattr(m,'closure',lambda:{str(Path(m.__file__)):m.sha(m.__file__)})
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    value=m.build(refs);out=tmp_path/'results'/(m.NAME+'_fixture.json');h=m.write(out,value)['sha256']
    assert m.verify(out,h)['verified'] and value['new_neural_calls']==value['new_physics_calls']==0
    with pytest.raises(FileExistsError):m.write(out,value)
    altered=copy.deepcopy(value);altered['comparisons'][0]['errors']['native7_distance']['C_worse']=False
    p=tmp_path/'results'/(m.NAME+'_tamper.json');th=m.write(p,altered)['sha256']
    with pytest.raises(ValueError):m.verify(p,th)
    Path(refs['C']['path']).write_bytes(b'{}')
    with pytest.raises(ValueError):m.build(refs)


def test_authority_wall_and_source_closure(tmp_path,monkeypatch):
    refs=fixture_files(tmp_path);monkeypatch.setattr(m,'actual_refs',lambda:refs)
    closure=m.closure();assert str(Path(m.__file__)) in closure
    assert all('/results/' not in p for p in closure)
    other=copy.deepcopy(refs);other['C']['sha256']='0'*64
    with pytest.raises(ValueError):m.build(other)
    tick=iter((0.,121.))
    with pytest.raises(ValueError):m.build(refs,clock=lambda:next(tick))


@pytest.mark.parametrize('data',[b'{"a":1,"a":2}',b'{"x":NaN}'])
def test_json_rejection(tmp_path,data):
    p=tmp_path/'bad.json';p.write_bytes(data)
    with pytest.raises(ValueError):m.read(p,m.sha(p))


def test_no_upstream_imports_or_fullstrict():
    tree=ast.parse(Path(m.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):assert not node.module.startswith(('experiments','torch','numpy'))
        if isinstance(node,ast.Import):assert not any(a.name.startswith(('experiments','torch','numpy')) for a in node.names)
        if isinstance(node,ast.Call):assert 'consume' not in ast.unparse(node.func)
