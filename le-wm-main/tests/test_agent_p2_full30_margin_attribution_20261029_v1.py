"""Fixed RAM full30 scalar ABI; no model/native/fullstrict imports."""
from copy import deepcopy
from fractions import Fraction
import math
import pytest
from experiments.spectral_reachability import agent_p2_full30_margin_attribution_20261029_v1 as m


def fixture(missing=False,tied=False):
    panel=[]
    for slot in range(4):
        for step in m.CP:
            pos=100.-step*(slot%2);vel=(step if tied else step*(2 if slot//2==0 else .5))*(slot%2)
            total=pos+vel
            values=dict(agent_xy_squared=pos-10,object_xy_squared=10.,joint_xy_squared=pos,raw_angle_squared=0.,
                velocity_squared=vel,native7_squared=total,joint_xy_error=math.sqrt(pos),object_xy_error=math.sqrt(10),
                native7_distance=math.sqrt(total),native_angle_term=0.,wrapped_angle_error=0.,native_success=False)
            absent=missing and slot==1 and step>=25
            panel.append(dict(slot=slot,root=slot//2,arm=m.ARMS[slot%2],control_step=step,absolute_step=10+step,
                checkpoint=None if absent else dict(step=step),metrics=None if absent else values,
                missing_reason='checkpoint_not_reached' if absent else None))
    pairs=[]
    for root in (0,1):
        for j,step in enumerate(m.CP):
            a,b=[panel[slot*7+j]['metrics'] for slot in (2*root,2*root+1)]
            delta=None if a is None or b is None else {k:b[k]-a[k] for k in a if k!='native_success'}
            parts=None if delta is None else {k:delta[k] for k in m.AXES}
            pairs.append(dict(root=root,control_step=step,direction='joint_two_round minus two_program_transfer',
                comparable=delta is not None,delta=delta,squared_components=parts,
                decomposition_roundoff=None if delta is None else delta[m.TOTAL]-math.fsum(parts.values()),
                native_success_delta=None if delta is None else 0))
    return dict(schema=m.SOURCE+'.summary',arguments={},own_sha256={},new_neural_calls=0,new_physics_calls=0,new_fullstrict_calls=0,
        summary=dict(denominator=dict(branches=4,known_roots=2,fresh_roots=0,checkpoint_positions=28,pairs=14),
            panel=panel,pairs=pairs,counts=dict(FLOPs=None,predictor=0),all_four_full30=not missing,**{k:False for k in m.FLAGS}))


def frac(x):return Fraction(int(x['numerator']),int(x['denominator']))


def test_fixed28cp14pair_and_sign_contributions():
    x=m.attribute(fixture())
    assert len(x['panel'])==28 and len(x['pairs'])==14
    assert x['margin_census'][m.TOTAL]==dict(negative=6,tie=2,positive=6,null=0)
    assert len(x['position_total_opposition_index'])==6
    assert x['original_counts']['FLOPs'] is None
    for p in x['pairs']:
        a=p['attribution']
        assert frac(a['exact_float_margin_minus_axis_sum'])==frac(a['exact_rounding_reconstruction'])
        if p['root']==0 and p['control_step']:
            assert a['axis_relation_to_total']['joint_xy_squared']=='opposed'
            assert a['axis_relation_to_total']['velocity_squared']=='aligned'


def test_missing_fixed_positions_null_not_terminal_fill():
    out=m.attribute(fixture(missing=True))
    assert len(out['pairs'])==14 and len(out['panel'])==28
    assert out['margin_census'][m.TOTAL]['null']==2
    assert sum(p['metrics'] is None for p in out['panel'])==2
    assert all(sum(c.values())==14 for c in out['margin_census'].values())


def test_total_tie_can_hide_nonzero_cancelling_axes():
    out=m.attribute(fixture(tied=True));assert out['margin_census'][m.TOTAL]['tie']==14
    assert out['position_total_opposition_index']==[]
    a=out['pairs'][1]['attribution'];assert a['axis_cancellation']
    assert a['axis_relation_to_total']['joint_xy_squared']==a['axis_relation_to_total']['velocity_squared']=='total_tie'


def test_fraction_isolates_nonzero_subtraction_and_reduction_roundoff():
    # Nonbinary coordinates make the TWO rounding mechanisms separately nonzero.
    a=fixture()['summary']['panel'][0]['metrics'];b=deepcopy(a)
    a.update(joint_xy_squared=.1,raw_angle_squared=.2,velocity_squared=.3,native7_squared=.6)
    b.update(joint_xy_squared=1.,raw_angle_squared=.4,velocity_squared=.5,native7_squared=1.9)
    delta={k:b[k]-a[k] for k in a if k!='native_success'}
    parts={k:delta[k] for k in m.AXES}
    saved=dict(delta=delta,native_success_delta=0,squared_components=parts,
        decomposition_roundoff=delta[m.TOTAL]-math.fsum(parts.values()))
    out=m.pair_math(a,b,saved)
    assert frac(out['exact_margin_reduction_residual'])!=0
    assert any(frac(v)!=0 for v in out['exact_subtraction_roundoff'].values())
    assert frac(out['exact_rounding_reconstruction'])==frac(out['exact_float_margin_minus_axis_sum'])


@pytest.mark.parametrize('kind',['schema','denominator','roworder','pairoder','arm','time','missing','margin','parts','roundoff',
    'native','flag','scalar','boolean','fullmissing'])
def test_mutations_fail_closed(kind):
    x=deepcopy(fixture());s=x['summary']
    if kind=='schema':x['schema']='old.summary'
    elif kind=='denominator':s['denominator']['pairs']=13
    elif kind=='roworder':s['panel'].reverse()
    elif kind=='pairoder':s['pairs'].reverse()
    elif kind=='arm':s['panel'][0]['arm']='joint'
    elif kind=='time':s['panel'][0]['absolute_step']=0
    elif kind=='missing':s['pairs'][0]['comparable']=False
    elif kind=='margin':s['pairs'][1]['delta'][m.TOTAL]+=1.
    elif kind=='parts':s['pairs'][1]['squared_components']['velocity_squared']+=1.
    elif kind=='roundoff':s['pairs'][1]['decomposition_roundoff']=1.
    elif kind=='native':s['pairs'][0]['native_success_delta']=1
    elif kind=='flag':s['freshness']=True
    elif kind=='scalar':s['panel'][0]['metrics'][m.TOTAL]+=1.
    elif kind=='boolean':s['panel'][0]['metrics']['velocity_squared']=False
    else:s['panel'][0].update(metrics=None,checkpoint=None,missing_reason='checkpoint_not_reached')
    with pytest.raises((ValueError,KeyError)):m.attribute(x)


def test_saved_cli_persistence_and_wholeverify_fixture(tmp_path,monkeypatch):
    src=tmp_path/'source.json';src.write_bytes(m.encode(fixture()))
    monkeypatch.setattr(m,'SOURCE_SHA',m.sha(src))
    out=tmp_path/(m.NAME+'_RAM.json')
    m.main(['analyze','--summary',str(src),'--summary-sha',m.sha(src),'--output',str(out)])
    m.main(['verify','--analysis',str(out),'--analysis-sha',m.sha(out)])
    with pytest.raises(ValueError):m.main(['analyze','--summary',str(src),'--summary-sha',m.sha(src),'--output',str(out)])
    with pytest.raises(ValueError):m.analyze(str(src),'0'*64)
    with pytest.raises(ValueError):m.absolute('relative.json')
    bad=tmp_path/'bad.json';bad.write_bytes(b'{"a":1,"a":2}')
    with pytest.raises(ValueError):m.read(str(bad),m.sha(bad))
