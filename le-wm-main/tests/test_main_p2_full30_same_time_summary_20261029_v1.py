from copy import deepcopy
import math
import pytest
from experiments.spectral_reachability import main_p2_full30_same_time_summary_20261029_v1 as m


def fixture(stop=False):
    paths=['/RAM/p','/RAM/r','/RAM/c']; hashes=['a'*64,'b'*64,'c'*64]
    p=dict(schema=m.SOURCE+'.prepare',plans=[],source_roots=[dict(goal=[0.]*7)]*2)
    r=dict(schema=m.SOURCE+'.report',status='full30_four_native_recorded_NOT_admitted',errors=[],fixed_denominator=4,known_roots=2,fresh_roots=0,
           execution_binding=dict(prepared=paths[0],prepare_sha256=hashes[0]),rows=[],counts={},**{k:False for k in m.FLAGS})
    for i in range(4):
        n=14 if stop and i==0 else 30
        state=lambda b:[float(100+i+b),0.,10.,0.,0.,float(i),0.]
        cp=[dict(step=b,state=state(b),snapshot=dict(state=state(b))) for b in m.CP if b<=n]
        plan=dict(native_goal=[0.]*7,planner_plan=dict(root_context={'history':'same root'}))
        p['plans'].append(plan)
        r['rows'].append(dict(index=i,root_slot=i//2,condition='off',arm=m.ARMS[i%2],plan=deepcopy(plan),
            status='native_stopped' if n<30 else 'horizon_completed',first_error=None,secondary_errors=[],goal=[0.]*7,
            own_returned=n,own_attempted=n,root_state=[100.,0.,10.,0.,0.,0.,0.],final_state=state(n),
            responses=[dict(step=b,state=state(b),terminated=b==n and n<30,truncated=False) for b in range(1,n+1)],
            checkpoints=cp,missing_checkpoints=[b for b in m.CP if b>n]))
        # cp0 is common physical root, regardless of branch index.
        r['rows'][-1]['checkpoints'][0].update(state=r['rows'][-1]['root_state'],snapshot=dict(state=r['rows'][-1]['root_state']))
    c=dict(schema=m.SOURCE+'.consume',**{k:deepcopy(v) for k,v in r.items() if k not in ('schema','execution_binding')},
           plans=deepcopy(p['plans']),physical_report_sha256=hashes[1],prepare_sha256=hashes[0],bindings={paths[0]:hashes[0],paths[1]:hashes[1]})
    return p,r,c,paths,hashes


def test_all28_fourteen_pairs():
    out=m.summarize(*fixture())
    assert len(out['panel'])==28 and len(out['pairs'])==14 and out['all_four_full30']
    assert all(x['comparable'] for x in out['pairs'])


def test_earlystop_cp30_null_not_terminal_substitution():
    out=m.summarize(*fixture(True))
    assert len(out['pairs'])==14 and not out['pairs'][-1]['delta'] is None
    assert out['pairs'][3]['delta'] is None and out['panel'][6]['metrics'] is None
    assert out['terminals'][0]['actual_control_step']==14


@pytest.mark.parametrize('fault',['schema','count','rows','sha','time','padding','task','flag'])
def test_mutations(fault):
    p,r,c,paths,hs=fixture()
    if fault=='schema':c['schema']='old.consume'
    elif fault=='count':c['fixed_denominator']=3
    elif fault=='rows':c['rows'].pop()
    elif fault=='sha':c['physical_report_sha256']='wrong'
    elif fault=='flag':c['P2_admission']=True
    else:
        if fault=='time':r['rows'][0]['checkpoints'][1]['state']=[0.]*7
        elif fault=='padding':r['rows'][0]['responses'][0]['terminated']=True
        else:r['rows'][1]['goal']=[1.]*7
        c['rows']=deepcopy(r['rows'])
    with pytest.raises(ValueError):m.summarize(p,r,c,paths,hs)


def test_fsum_metric_and_original_angle_boundary():
    x=m.scalars([3.,4.,0.,0.,0.,0.,12.],[0.]*7)
    assert x['native7_distance']==13 and x['joint_xy_squared']==25 and x['velocity_squared']==144
    y=m.scalars([0.,0.,0.,0.,8.,0.,0.],[0.]*7)
    assert y['native_angle_term']<0 and y['native_success']
    assert math.isclose(y['wrapped_angle_error'],8-2*math.pi)


def test_no_bool_or_relative_path():
    with pytest.raises(ValueError):m.scalars([False]*7,[0.]*7)
    with pytest.raises(ValueError):m.absolute('relative.json')
