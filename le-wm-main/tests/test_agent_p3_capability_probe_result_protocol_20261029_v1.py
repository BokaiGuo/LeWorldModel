"""Pre-result synthetic RAM only. No real reports, physical calls or model IO."""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p3_capability_probe_result_protocol_20261029_v1 as m


def fixture():
    old=dict(schema=m.CAPABILITY+'.summary',signals=[{'RAM_only':i} for i in range(7)],probes=[],comparisons=[],
        C_dual_control_conditions=[],P1_admission=False,P3_admission=False,scientific_admission=False,full_C3_demonstrated=False)
    for i,(root,role) in enumerate(m.ENDPOINTS):
        old['probes'].append(dict(index=i,root_slot=root,role=role,native_cp10=dict(goal7=[0.]*7)))
    for ci,(root,arm) in enumerate(m.LOGICAL):
        for j,role in enumerate(('Z','P')):
            old['comparisons'].append(dict(index=2*ci+j,root_slot=root,C_arm=arm,control_role=role,available=ci==0,
                immediate_cost_condition=None if ci else bool(j),
                R10_original_solver_C_vs_control=dict(relation='null' if ci else 'less',a_minus_b=None if ci else -.1)))
        old['C_dual_control_conditions'].append(dict(root_slot=root,C_arm=arm,C_immediate_cost_both_controls=None if ci else False))
    values=[(1.,4.),(2.,3.),(2.,3.),(5.,6.),(7.,7.)]
    rows=[]
    for index,(endpoint,root,role,probe) in enumerate(m.JOBS):
        x=values[endpoint][probe=='Z2'];s=[x]+[0.]*6
        rows.append(dict(index=index,endpoint=endpoint,root_slot=root,role=role,probe=probe,status='horizon_completed',
            goal7=[0.]*7,cp15_state7=s,cp15_native_reward=-x,cp15_native_success=True,
            terminal_control_step=15,terminal_state7=s,terminal_goal7=[0.]*7,terminal_native_success=True,failure=None))
    return rows,old


def test_fixed_full10_5pairs_8logical_no_witness_and_known_false():
    rows,old=fixture();out=m.evaluate(rows,old)
    assert len(out['branches'])==10 and len(out['U_Z2_pairs'])==5 and len(out['C_control_pairs'])==8
    assert [x['G'] for x in out['U_Z2_pairs']]==[3.,1.,1.,1.,0.]
    assert out['U_Z2_pairs'][4]['G_relation_to_zero']=='tie'
    assert [p['full_chain'] for p in out['C_control_pairs']]==[False,True]+[None]*6
    assert out['C_dual_control_conditions'][0]['original_immediate_cost_both_controls'] is False
    assert out['C_dual_control_conditions'][0]['full_chain_both_controls'] is False
    assert all(p['original_C_no_witness'] for p in out['C_control_pairs'][2:])
    assert all(p['G_C_vs_control']['relation']=='null' for p in out['C_control_pairs'][2:])
    assert all(out[k] is False for k in m.FLAGS)
    assert not out['authentication_performed'] and not out['production_consumer_ready']


def test_native_all_axes_velocity_and_object_not_only_joint():
    x=m.native_math([3.,4.,12.,0.,.5,8.,15.],[0.]*7)
    assert x['joint_xy_error']==13. and x['object_xy_error']==12. and x['agent_xy_error']==5.
    assert x['velocity_error']==17. and x['native7_squared']==458.25
    assert x['native7_squared']==math.fsum((x['joint_xy_squared'],x['raw_angle_squared'],x['velocity_squared']))
    assert x['raw_angle_error']==.5 and x['group_sum_roundoff']==0.


def test_direction_primary_original_reward_and_fsum_separate():
    rows,old=fixture()
    # An explicit arithmetic discrepancy must remain visible, not redefine native reward.
    rows[0]['cp15_native_reward']=-1.0000000000000002
    x=m.evaluate(rows,old)
    assert x['branches'][0]['D_original_reward']==1.0000000000000002
    assert x['branches'][0]['metrics']['native7_distance']==1.
    assert x['branches'][0]['D_fsum_minus_original']<0.
    assert x['U_Z2_pairs'][0]['G']==4.-1.0000000000000002
    assert x['U_Z2_pairs'][0]['fsum_G']==3.


def test_success_is_original_label_not_replaced_by_formula_or_distance():
    rows,old=fixture();rows[0]['cp15_native_success']=rows[0]['terminal_native_success']=False
    x=m.evaluate(rows,old)
    assert x['branches'][0]['original_native_success'] is False
    assert x['branches'][0]['metrics']['native_success_formula_diagnostic'] is True
    assert x['branches'][0]['native_success_formula_matches_original'] is False
    assert x['U_Z2_pairs'][0]['success_relation']=='Z2_only'
    assert x['U_Z2_pairs'][0]['G']>0.


def test_raw_angle_historical_native_bug_preserved_and_wrap_separate():
    x=m.native_math([0.,0.,0.,0.,7.,100.,100.],[0.]*7)
    assert x['native_angle_term']<0. and x['native_success_formula_diagnostic'] is True
    assert x['wrapped_success_diagnostic'] is False


def test_agent_velocity_uses_goal_velocity_not_zero():
    x=m.native_math([0.,0.,0.,0.,0.,3.,4.],[0.,0.,0.,0.,0.,3.,4.])
    assert x['velocity_error']==0. and x['native7_distance']==0.


def test_native_earlystop_keeps_terminal_and_cp15_null():
    rows,old=fixture();r=rows[0]
    r.update(status='native_stopped',cp15_state7=None,cp15_native_reward=None,cp15_native_success=None,terminal_control_step=14)
    x=m.evaluate(rows,old)
    assert x['branches'][0]['metrics'] is None
    assert x['branches'][0]['terminal_metrics'] is not None
    assert x['branches'][0]['terminal_control_step']==14
    assert x['U_Z2_pairs'][0]['G'] is None
    assert all(v['relation']=='null' for v in x['U_Z2_pairs'][0]['error_improvements'].values())
    assert x['C_control_pairs'][0]['full_chain'] is False  # known false is not erased by pending operands
    assert x['C_control_pairs'][1]['full_chain'] is None


def test_native_stop_exact15_is_real_endpoint():
    rows,old=fixture();rows[0]['status']='native_stopped'
    assert m.evaluate(rows,old)['branches'][0]['eligible_same_time_endpoint'] is True


def test_engineering_failure_after15_retains_observation_but_no_valid_gain():
    rows,old=fixture();rows[0].update(status='engineering_failed',failure={'cleanup':'failed'})
    x=m.evaluate(rows,old)
    assert x['branches'][0]['metrics'] is not None
    assert x['branches'][0]['reported_D_original_reward']==1.
    assert not x['branches'][0]['eligible_same_time_endpoint']
    assert x['branches'][0]['D_original_reward'] is None
    assert x['U_Z2_pairs'][0]['G'] is None
    assert x['U_Z2_pairs'][0]['success_relation']=='null'
    assert x['branches'][0]['failure']=={'cleanup':'failed'}


def test_failure_before_any_return_has_no_fabricated_terminal():
    rows,old=fixture();rows[0].update(status='transport_failed',cp15_state7=None,cp15_native_success=None,
        cp15_native_reward=None,terminal_control_step=0,terminal_state7=None,terminal_goal7=None,
        terminal_native_success=None,failure={'intent_lost':True})
    out=m.evaluate(rows,old)
    assert out['branches'][0]['metrics'] is out['branches'][0]['terminal_metrics'] is None
    assert len(out['branches'])==10 and out['U_Z2_pairs'][0]['G'] is None


def test_terminal_goal_before_attachment_must_not_be_assumed_task_goal():
    rows,old=fixture();rows[0].update(status='native_stopped',terminal_control_step=0,
        cp15_state7=None,cp15_native_success=None,cp15_native_reward=None,terminal_goal7=[1.]+[0.]*6)
    x=m.evaluate(rows,old)['branches'][0]
    assert x['terminal_metrics']['native7_distance']==0.
    assert x['metrics'] is None


@pytest.mark.parametrize('terms,answer',[([False,None],False),([True,None],None),([True,True],True),([True,False],False),([None,None],None)])
def test_three_valued_conjunction(terms,answer):
    assert m.conjunction(terms) is answer


@pytest.mark.parametrize('a,b,relation',[(0.,-0.,'tie'),(5e-324,-5e-324,'greater'),(-5e-324,5e-324,'less'),(None,1.,'null')])
def test_strict_sign_no_products_tie_null(a,b,relation):
    assert m.compare(a,b)['relation']==relation


@pytest.mark.parametrize('damage',['count','order','role','endpoint','missinglabel','fake15','missing_complete','goal','flag','borrow_C','prior_conjunction'])
def test_fail_closed_fixed_denominator_no_relabeling(damage):
    rows,old=fixture()
    if damage=='count':rows.pop()
    elif damage=='order':rows.reverse()
    elif damage=='role':rows[0]['role']='P'
    elif damage=='endpoint':rows[0]['endpoint']=1
    elif damage=='missinglabel':rows[0]['cp15_state7']=None
    elif damage=='fake15':rows[0]['terminal_control_step']=14
    elif damage=='missing_complete':rows[0].update(cp15_state7=None,cp15_native_success=None,cp15_native_reward=None)
    elif damage=='goal':rows[0]['goal7'][0]=1.
    elif damage=='flag':old['P3_admission']=True
    elif damage=='borrow_C':old['comparisons'][2]['immediate_cost_condition']=False
    else:old['C_dual_control_conditions'][0]['C_immediate_cost_both_controls']=True
    with pytest.raises(ValueError):m.evaluate(rows,old)


@pytest.mark.parametrize('bad',[[0.]*6,[True]+[0.]*6,[float('nan')]+[0.]*6,[float('inf')]+[0.]*6])
def test_bad_native_arrays_rejected(bad):
    with pytest.raises(ValueError):m.native_math(bad,[0.]*7)


def test_describe_cli_no_IO_or_producer_imports():
    tree=ast.parse(Path(m.__file__).read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.ImportFrom):assert not n.module.startswith(('experiments','numpy','torch','pymunk'))
        if isinstance(n,ast.Import):assert not any(v.name.startswith(('experiments','numpy','torch','pymunk')) for v in n.names)
        if isinstance(n,ast.Call):assert ast.unparse(n.func) not in ('open','eval','exec','__import__')
    run=subprocess.run([sys.executable,'-B','-m','experiments.spectral_reachability.'+m.NAME,'describe'],
                       cwd=Path(__file__).resolve().parents[1],check=True,capture_output=True,text=True)
    value=json.loads(run.stdout)
    assert value['denominators']==dict(endpoints=5,branches=10,U_Z2_pairs=5,C_logical_slots=4,C_no_witness=3,C_control_pairs=8)
    assert all(value[k] is False for k in m.FLAGS)
    assert value['production_consumer_ready'] is False
