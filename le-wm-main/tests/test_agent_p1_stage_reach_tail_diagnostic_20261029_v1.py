import json
import sys
from copy import deepcopy
from fractions import Fraction

import pytest

from experiments.spectral_reachability import agent_p1_stage_reach_tail_diagnostic_20261029_v1 as m


def panel(family, i, n):
    rows = [dict(index=j, stage_cost=1.+j, own_reach_value=2., original_objective=3.+j,
                 nonlinear_terminal_Q=1.+j) for j in range(n)]
    return dict(family=family, index=i, root_slot=i//2, condition=('off','on')[i%2],
        candidate_denominator=n, candidates=rows, status='complete_model_only', terminal_available=n,
        zero_prefix_index=0, literal_forward_requests=[], initial_Q=1.5, stage_greedy_index=0,
        original_stage_plus_ridge=dict(index=0,score=3.), nonlinear_terminal_Q=dict(index=0,score=1.))


def fixture():
    return dict(schema='agent_p1_terminal_witness_selector_20261028_v1.analysis',
        status='new_exploratory_model_only_selector_not_admitted', candidate_denominator=1056,
        P1_admission=False, panels=[panel(f,i,n) for f,n in (('allmode',42),('mixedmode',222)) for i in range(4)])


def tail():
    return [dict(index=i,endpoint=i//2,root=m.ENDPOINTS[i//2][0],role=m.ENDPOINTS[i//2][1],
        kind=('U','Z2')[i%2], status='horizon_completed', goal7=[0.,0.,0.,0.,0.,60.,-180.],
        cp15=[1.,2.,3.,4.,0.,59.,-180.], cp20=[1.,2.,3.,4.,0.,0.,0.]) for i in range(10)]


def test_fixed_p1_all_slots_and_exact_decomposition():
    out=m.p1_breakpoints(fixture())
    assert len(out['panels'])==8 and sum(x['candidate_denominator'] for x in out['panels'])==1056
    x=out['panels'][0]['candidates'][1]
    assert Fraction(x['stage_delta'])==1 and Fraction(x['reach_delta'])==0
    assert Fraction(x['exact_sum_delta'])==1 and x['real_benefit'] is None
    assert out['upstream_fullstrict_rebuilt'] is False


def test_stage_reach_disagreement_and_native_winner():
    d=fixture(); p=d['panels'][0]; x=p['candidates'][1]
    x.update(stage_cost=2.,own_reach_value=.5,original_objective=2.5,nonlinear_terminal_Q=.1)
    p['original_stage_plus_ridge'].update(index=1,score=2.5)
    p['nonlinear_terminal_Q'].update(index=1,score=.1)
    result=m.p1_breakpoints(d)['panels'][0]
    assert result['selected_index']==1 and result['stage_greedy_index']==0
    assert Fraction(result['candidates'][0]['stage_delta'])==-1
    assert Fraction(result['candidates'][0]['reach_delta'])==Fraction(3,2)


def test_native_float_rounding_not_silently_exact():
    d=fixture(); p=d['panels'][0]; p['candidates'][1].update(stage_cost=3.,own_reach_value=.1,original_objective=3.+.1)
    x=m.p1_breakpoints(d)['panels'][0]['candidates'][1]
    assert Fraction(x['objective_rounding_delta']) != 0


@pytest.mark.parametrize('mutate', [lambda d:d['panels'].pop(),
    lambda d:d['panels'][0]['candidates'].pop(),
    lambda d:d['panels'][0]['candidates'][0].update(original_objective=999),
    lambda d:d['panels'][0].update(stage_greedy_index=1),
    lambda d:d['panels'][0].update(terminal_available=41),
    lambda d:d['panels'][0]['candidates'][1].update(index=0)])
def test_p1_tampering_rejected(mutate):
    d=fixture(); mutate(d)
    with pytest.raises(ValueError): m.p1_breakpoints(d)


def test_nonzero_goal_velocity_settling_can_increase_error():
    out=m.tail_ram(tail()); row=out['rows'][0]
    assert row['cp15']['agent_velocity']==1
    assert row['cp20']['agent_velocity']>row['cp15']['agent_velocity']
    assert row['cp20']['native7_distance']>row['cp15']['native7_distance']
    assert row['cp20']['native_success'] is True  # success excludes velocity
    assert len(out['pairs'])==5 and out['branch_denominator']==10
    assert out['pairs'][0]['cp20']['agent_velocity']['relation']=='tie'
    assert out['source_authenticated'] is False and out['physical_run'] is False


def test_native_angle_unchanged_wrapped_only_diagnostic():
    x=m.metrics([0,0,0,0,7,0,0],[0,0,0,0,0,0,0])
    assert x['native_success'] is True and x['raw_angle']==7 and x['wrapped_angle']<1


def test_earlystop_null_retains_both_denominators():
    rows=tail(); rows[0].update(status='native_stopped',cp20=None)
    out=m.tail_ram(rows)
    assert out['rows'][0]['cp20'] is None
    assert out['pairs'][0]['cp20']['native7_distance']['relation']=='null'
    assert out['logical_C']==4 and out['logical_C_comparisons']==8 and out['inherited_no_witness']==3


def test_nonzero_goal_bits_must_match():
    rows=tail(); rows[1]['goal7'][-1]=-179.
    with pytest.raises(ValueError,match='same goal'): m.tail_ram(rows)
    rows=tail(); rows[1]['goal7'][0]=-0.
    with pytest.raises(ValueError,match='same goal'): m.tail_ram(rows)


def test_no_missing_or_reordered_or_fake_completed():
    rows=tail()
    with pytest.raises(ValueError): m.tail_ram(rows[:-1])
    rows[0],rows[1]=rows[1],rows[0]
    with pytest.raises(ValueError): m.tail_ram(rows)
    rows=tail(); rows[0]['cp20']=None
    with pytest.raises(ValueError): m.tail_ram(rows)


def test_tempdisk_build_verify_exclusive_and_tampering(tmp_path,monkeypatch):
    p=tmp_path/'fixture.json'; p.write_bytes(m.encode(fixture()))
    monkeypatch.setattr(m,'P1',p); monkeypatch.setattr(m,'P1_SHA',m.digest(p))
    out=tmp_path/'output.json'
    monkeypatch.setattr(sys,'argv',['prog','analyze','--output',str(out)]); m.main()
    monkeypatch.setattr(sys,'argv',['prog','verify','--output',str(out)]); m.main()
    monkeypatch.setattr(sys,'argv',['prog','analyze','--output',str(out)])
    with pytest.raises(FileExistsError): m.main()
    obj=json.loads(out.read_bytes()); obj['candidate_denominator']=1; out.write_bytes(m.encode(obj))
    monkeypatch.setattr(sys,'argv',['prog','verify','--output',str(out)])
    with pytest.raises(ValueError): m.main()


def test_bad_pin_rejected(tmp_path,monkeypatch):
    p=tmp_path/'fixture.json'; p.write_bytes(m.encode(fixture()))
    monkeypatch.setattr(m,'P1',p); monkeypatch.setattr(m,'P1_SHA','0'*64)
    with pytest.raises(ValueError,match='pinned'): m.build()
