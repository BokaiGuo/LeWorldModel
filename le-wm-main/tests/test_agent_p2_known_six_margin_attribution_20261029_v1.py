"""RAM algebra fixtures only; no physics, neural or fullstrict calls."""
from copy import deepcopy
from fractions import Fraction
import pytest
from experiments.spectral_reachability import agent_p2_known_six_margin_attribution_20261029_v1 as m


def fixture(missing=False):
    rows=[]
    for slot in range(6):
        for step in m.CP:
            # Position improves, velocity worsens; root0 total worsens, root1 improves.
            p=100.-10*(slot%3) if step else 100.
            v=(20. if slot//3==0 else 5.)*(slot%3) if step else 0.
            a=0.;total=p+a+v
            metrics=dict(joint_xy_squared=p,raw_angle_squared=a,agent_velocity_squared=v,native7_squared=total,
                joint_xy_error=p**.5,native7_distance=total**.5,native_success=False)
            absent=missing and slot==1 and step==15
            rows.append(dict(slot=slot,root_slot=slot//3,arm=m.ARMS[slot%3],step=step,
                metrics=None if absent else metrics,missing_reason='checkpoint_not_reached' if absent else None))
    pairs=[]
    for root in (0,1):
        for l,r in ((0,1),(0,2),(1,2)):
            for step in m.CP:
                left,right=3*root+l,3*root+r;a,b=[rows[i*4+m.CP.index(step)]['metrics'] for i in (left,right)]
                known=a is not None and b is not None
                pairs.append(dict(root_slot=root,step=step,left_slot=left,right_slot=right,comparable=known,
                    missing_slots=[i for i,x in ((left,a),(right,b)) if x is None],
                    delta_right_minus_left=None if not known else {k:b[k]-a[k] for k in ('native7_squared','native7_distance','joint_xy_error',*m.AXES)}))
    return dict(schema=m.SOURCE+'.summary',new_neural_calls=0,new_physics_calls=0,new_fullstrict_calls=0,bindings={},summary=dict(
        rows=rows,pairs=pairs,denominator=dict(slots=6,checkpoints=24,pairs=24),known_roots=2,fresh_roots=0,
        comparable_pairs=sum(x['comparable'] for x in pairs),**{k:False for k in m.FLAGS}))


def test_all24_cp_and_pair_contribution_no_selection():
    a=m.attribute(fixture())
    assert len(a['rows'])==len(a['pairs'])==24
    assert a['total_margin_census']==dict(negative=9,tie=6,positive=9,null=0)
    assert len(a['position_total_opposition_index'])==9
    for row in a['pairs']:
        p=row['attribution']
        if row['step']==0:
            assert p['signs']['native7_squared']=='tie' and set(p['axis_relation_to_total'].values())=={'tie'}
        elif row['root_slot']==0:
            assert p['position_vs_total_squared_opposite'] and p['position_vs_total_distance_opposite']
            assert p['axis_relation_to_total']['joint_xy_squared']=='opposed'
            assert p['axis_relation_to_total']['agent_velocity_squared']=='aligned'


def test_null_keeps_fixed_denominator():
    a=m.attribute(fixture(True));assert len(a['pairs'])==24
    assert a['total_margin_census']['null']==2
    assert all(x['attribution'] is None for x in a['pairs'] if not x['comparable'])
    assert all(sum(c.values())==24 and c['null']==2 for c in a['axis_margin_census'].values())


def test_total_tie_with_nonzero_opposed_axes_is_not_zero_response():
    x=fixture();s=x['summary']
    for row in s['rows']:
        v=row['metrics'];v['agent_velocity_squared']=100.-v['joint_xy_squared']
        v['native7_squared']=100.;v['native7_distance']=10.
    for pair in s['pairs']:
        a,b=[s['rows'][slot*4+m.CP.index(pair['step'])]['metrics'] for slot in (pair['left_slot'],pair['right_slot'])]
        pair['delta_right_minus_left']={k:b[k]-a[k] for k in pair['delta_right_minus_left']}
    out=m.attribute(x)
    assert out['total_margin_census']['tie']==24 and out['position_total_opposition_index']==[]
    p=out['pairs'][1]['attribution']
    assert p['axis_cancellation'] and p['axis_relation_to_total']['joint_xy_squared']=='total_tie'
    assert p['axis_relation_to_total']['agent_velocity_squared']=='total_tie'


@pytest.mark.parametrize('kind',['denominator','order','pairorder','margin','scalar','missing','count','flag','schema','boolean'])
def test_failclosed_mutations(kind):
    x=deepcopy(fixture());s=x['summary']
    if kind=='denominator':s['denominator']['pairs']=23
    elif kind=='order':s['rows'][0]['slot']=5
    elif kind=='pairorder':s['pairs'].reverse()
    elif kind=='margin':s['pairs'][1]['delta_right_minus_left']['native7_squared']+=1.
    elif kind=='scalar':s['rows'][1]['metrics']['native7_squared']+=1.
    elif kind=='missing':s['pairs'][0]['missing_slots']=[1]
    elif kind=='count':s['comparable_pairs']=23
    elif kind=='flag':s['freshness']=True
    elif kind=='schema':x['schema']='other'
    else:s['rows'][0]['metrics']['joint_xy_squared']=True
    with pytest.raises((ValueError,KeyError)):m.attribute(x)


def test_exact_scalar_rounding_not_relabelled_as_velocity():
    x=fixture();row=x['summary']['rows'][1]['metrics']
    row.update(joint_xy_squared=.1,raw_angle_squared=.2,agent_velocity_squared=.3,native7_squared=.6000000000000001,
        joint_xy_error=.1**.5,native7_distance=.6000000000000001**.5)
    c=m.checked_metrics(row)['exact_scalar_sum_residual']
    assert Fraction(int(c['numerator']),int(c['denominator']))!=0


def test_saved_write_and_wholeverify(tmp_path):
    src=tmp_path/'source.json';src.write_bytes(m.encode(fixture()))
    out=tmp_path/(m.NAME+'_RAM.json')
    m.main(['analyze','--summary',str(src),'--summary-sha',m.sha(src),'--output',str(out)])
    m.main(['verify','--analysis',str(out),'--analysis-sha',m.sha(out)])
    with pytest.raises(ValueError):m.main(['analyze','--summary',str(src),'--summary-sha',m.sha(src),'--output',str(out)])
    with pytest.raises(ValueError):m.absolute('relative.json')
