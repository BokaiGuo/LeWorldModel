"""Stage C pre-result arithmetic kernel, NOT a production saved consumer.

No producer imports or file IO. MAIN's future actual receipt mapping/authority
must be supplied separately after its final ABI and fullstrict are available.
The describe CLI fixes the analysis contract; evaluate is a RAM-testable kernel.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math

NAME = 'agent_p3_capability_probe_result_protocol_20261029_v1'
CAPABILITY = 'agent_p3_capability_signal_summary_20261029_v1'
ENDPOINTS = ((0,'C'),(0,'Z'),(0,'P'),(1,'Z'),(1,'P'))
JOBS = tuple((i,root,role,probe) for i,(root,role) in enumerate(ENDPOINTS) for probe in ('U','Z2'))
LOGICAL = ((0,'H1'),(0,'H3'),(1,'H1'),(1,'H3'))
AXIS_ERRORS = ('native7_distance','native7_squared','joint_xy_error','joint_xy_squared',
    'agent_xy_error','agent_xy_squared','object_xy_error','object_xy_squared',
    'velocity_error','velocity_squared','raw_angle_error','raw_angle_squared','wrapped_angle_error')
FLAGS = dict(P1_admission=False,P2_admission=False,P3_admission=False,full_C3_demonstrated=False,
    scientific_admission=False,freshness=False,independent_confirmation=False,physical_release=False,
    equal_FLOPs=False,equal_total_compute=False,training=False)


def need(ok,message):
    if not ok:
        raise ValueError(message)


def finite(value):
    need(type(value) in (float,int) and math.isfinite(value),'finite scalar, no bool')
    return float(value)


def vector7(value):
    need(type(value) is list and len(value)==7,'literal native seven-vector')
    return [finite(v) for v in value]


def native_math(state,goal):
    s,g=vector7(state),vector7(goal)
    d=[a-b for a,b in zip(s,g,strict=True)]
    squares=[x*x for x in d]
    need(all(math.isfinite(x) for x in squares),'finite native residual arithmetic')
    total=math.fsum(squares);xy=math.fsum(squares[:4]);agent=math.fsum(squares[:2])
    obj=math.fsum(squares[2:4]);vel=math.fsum(squares[5:]);angle=abs(d[4])
    # Preserve the historical native angle behavior; wrap is diagnostic only.
    native_angle=min(angle,2*math.pi-angle)
    wrapped=abs(math.atan2(math.sin(d[4]),math.cos(d[4])))
    return dict(state7=s,goal7=g,residual7=d,squared_residual7=squares,
        native7_distance=math.sqrt(total),native7_squared=total,reward_formula=-math.sqrt(total),
        joint_xy_error=math.sqrt(xy),joint_xy_squared=xy,agent_xy_error=math.sqrt(agent),agent_xy_squared=agent,
        object_xy_error=math.sqrt(obj),object_xy_squared=obj,velocity_error=math.sqrt(vel),velocity_squared=vel,
        raw_angle_error=angle,raw_angle_squared=squares[4],wrapped_angle_error=wrapped,
        native_angle_term=native_angle,negative_native_angle=native_angle<0,
        native_success_formula_diagnostic=math.sqrt(xy)<20 and native_angle<math.pi/9,
        wrapped_success_diagnostic=math.sqrt(xy)<20 and wrapped<math.pi/9,
        group_sum_roundoff=total-math.fsum((xy,squares[4],vel)))


def compare(a,b):
    if a is None or b is None:
        return dict(A_minus_B=None,relation='null')
    a,b=finite(a),finite(b)
    return dict(A_minus_B=a-b,relation='less' if a<b else 'greater' if a>b else 'tie')


def conjunction(terms):
    need(all(v is None or type(v) is bool for v in terms),'three-valued predicates')
    if any(v is False for v in terms):
        return False
    return None if any(v is None for v in terms) else True


def protocol():
    return dict(name=NAME,scope='known2 pre-result StageC analysis contract; no production consumer yet',
        endpoints=[dict(index=i,root_slot=r,role=a) for i,(r,a) in enumerate(ENDPOINTS)],
        jobs=[dict(index=i,endpoint=e,root_slot=r,role=a,probe=p) for i,(e,r,a,p) in enumerate(JOBS)],
        native_axis_order=['agent_x','agent_y','object_x','object_y','object_angle_mod2pi','agent_vx','agent_vy'],
        fixed_checkpoints=[0,5,10,15],same_time_endpoint=dict(control=15,absolute=25),
        error_columns=list(AXIS_ERRORS),G='D15(Z2)-D15(U); positive favors U; original native reward-distance primary',
        secondary_G='each fixed error(Z2)-error(U), separate units; no combined score/winner',
        chain=['D10(C)>D10(B)','R10(C)<R10(B)','G(C)>G(B)','G(C)>0','D15(C,U)<D15(B,U)'],
        C_controls=['Z','P'],logical_C_slots=[list(x) for x in LOGICAL],
        denominators=dict(endpoints=5,branches=10,U_Z2_pairs=5,C_logical_slots=4,C_no_witness=3,C_control_pairs=8),
        null_policy='missing cp15 remains null; actual native terminal separately retained, no padding or endpoint substitution',
        independent_new_roots=0,known_roots=2,authentication_performed=False,production_consumer_ready=False,
        future_authority_required=['actual Pauli source ABI and SHA','actual ten-job report/batch and fullreceipt SHAs',
                                   'MAIN capability summary actual SHA'],**FLAGS)


def validate_row(row,index):
    e,r,a,p=JOBS[index]
    need((row['index'],row['endpoint'],row['root_slot'],row['role'],row['probe'])==(index,e,r,a,p),
         'all ten fixed branch identities/order')
    need(row['status'] in ('horizon_completed','native_stopped','engineering_failed','transport_failed'),
         'explicit complete/earlystop/failure, never silently dropped')
    vector7(row['goal7'])
    last=row['terminal_control_step']
    need(type(last) is int and 0<=last<=15,'actual terminal time, not padding')
    if row['terminal_state7'] is not None:
        vector7(row['terminal_state7'])
        vector7(row['terminal_goal7'])
        need(type(row['terminal_native_success']) is bool,'original terminal native label')
    else:
        need(row['terminal_native_success'] is row['terminal_goal7'] is None,'missing terminal is not false')
    if row['cp15_state7'] is None:
        need(row['cp15_native_success'] is row['cp15_native_reward'] is None,'missing cp15 labels null, not 0')
    else:
        vector7(row['cp15_state7'])
        need(last==15 and
             type(row['cp15_native_success']) is bool and row['cp15_state7']==row['terminal_state7'] and
             row['cp15_native_success'] is row['terminal_native_success'] and
             row['terminal_goal7']==row['goal7'],'real returned cp15 under original task goal only')
        reward=finite(row['cp15_native_reward']);need(reward<=0,'original native negative-distance reward')
    if row['status']=='horizon_completed':
        need(row['cp15_state7'] is not None and last==15,'complete full15 has actual checkpoint')


def evaluate(rows,capability):
    """Pure math only. These input records are NOT certificates/receipt wrappers.

    A future saved reader must authenticate the complete actual Pauli receipt,
    map native returns and all guards/raw/history, and retain all failures FIRST.
    This function never certifies those preconditions or releases actions.
    """
    need(type(rows) is list and len(rows)==10,'fixed ten branch denominator')
    need(capability['schema']==CAPABILITY+'.summary' and len(capability['probes'])==5 and
         len(capability['signals'])==7 and len(capability['comparisons'])==8 and
         len(capability['C_dual_control_conditions'])==4,'complete prior seven/five/eight summary ABI')
    need(all(capability[k] is False for k in ('P1_admission','P3_admission','scientific_admission','full_C3_demonstrated')),
         'prior scientific gates stay false')
    branches=[]
    for i,row in enumerate(rows):
        validate_row(row,i)
        prior=capability['probes'][row['endpoint']]
        need((prior['index'],prior['root_slot'],prior['role'])==(row['endpoint'],row['root_slot'],row['role']) and
             row['goal7']==prior['native_cp10']['goal7'],'same original root/role/native goal')
        m=None if row['cp15_state7'] is None else native_math(row['cp15_state7'],row['goal7'])
        terminal=None if row['terminal_state7'] is None else native_math(row['terminal_state7'],row['terminal_goal7'])
        eligible=m is not None and row['status'] in ('horizon_completed','native_stopped')
        reported_D=None if row['cp15_native_reward'] is None else -row['cp15_native_reward']
        original_D=reported_D if eligible else None
        branches.append(dict(index=i,endpoint=row['endpoint'],root_slot=row['root_slot'],role=row['role'],probe=row['probe'],
            status=row['status'],control_step=15,absolute_step=25,metrics=m,
            original_native_success=row['cp15_native_success'],eligible_same_time_endpoint=eligible,
            D_original_reward=original_D,reported_D_original_reward=reported_D,
            D_fsum_minus_original=None if m is None else m['native7_distance']-reported_D,
            native_success_formula_matches_original=None if m is None else m['native_success_formula_diagnostic'] is row['cp15_native_success'],
            terminal_control_step=row['terminal_control_step'],terminal_metrics=terminal,
            terminal_native_success=row['terminal_native_success'],failure=deepcopy(row['failure'])))
    gains=[]
    for i,(root,role) in enumerate(ENDPOINTS):
        u,z=branches[2*i:2*i+2]
        errors={k:compare(None if not z['eligible_same_time_endpoint'] else z['metrics'][k],
                          None if not u['eligible_same_time_endpoint'] else u['metrics'][k]) for k in AXIS_ERRORS}
        primary=compare(z['D_original_reward'],u['D_original_reward'])
        gains.append(dict(index=i,root_slot=root,role=role,U_job=2*i,Z2_job=2*i+1,
            G=primary['A_minus_B'],G_relation_to_zero='null' if primary['A_minus_B'] is None else compare(primary['A_minus_B'],0.)['relation'],
            fsum_G=errors['native7_distance']['A_minus_B'],error_improvements=errors,
            original_success_U=u['original_native_success'],original_success_Z2=z['original_native_success'],
            success_relation='null' if not u['eligible_same_time_endpoint'] or not z['eligible_same_time_endpoint'] else
                'tie' if u['original_native_success'] is z['original_native_success'] else 'U_only' if u['original_native_success'] else 'Z2_only',
            source_statuses=[u['status'],z['status']]))
    pairs=[]
    for ci,(root,arm) in enumerate(LOGICAL):
        for role in ('Z','P'):
            pi=len(pairs);old=capability['comparisons'][pi]
            need((old['index'],old['root_slot'],old['C_arm'],old['control_role'])==(pi,root,arm,role),
                 'all original logical C/control identities')
            bi=ENDPOINTS.index((root,role));b=gains[bi];c=gains[0] if ci==0 else None
            immediate=old['immediate_cost_condition']
            need(immediate is None or type(immediate) is bool,'original immediate condition, not recomputed/overwritten')
            R=old['R10_original_solver_C_vs_control']['relation']
            need(R in ('less','greater','tie','null'),'original solver comparison')
            need(old['available'] is (ci==0) and (ci==0 or (immediate is None and R=='null')),
                 'three original C no_witness keep all comparison operands null')
            gcompare=compare(None if c is None else c['G'],b['G'])
            dcompare=compare(None if c is None else branches[0]['D_original_reward'],branches[2*bi]['D_original_reward'])
            legs=dict(immediate_cost=immediate,original_solver_R=None if R=='null' else R=='less',
                G_C_greater_control=None if gcompare['relation']=='null' else gcompare['relation']=='greater',
                G_C_positive=None if c is None or c['G'] is None else c['G']>0,
                final_CU_better_controlU=None if dcompare['relation']=='null' else dcompare['relation']=='less')
            pairs.append(dict(index=pi,root_slot=root,C_arm=arm,control_role=role,original_C_no_witness=ci!=0,
                original_immediate_cost_condition=immediate,original_solver_R_C_vs_control=deepcopy(old['R10_original_solver_C_vs_control']),
                G_C_vs_control=gcompare,D15_CU_vs_controlU=dcompare,conditions=legs,
                full_chain=conjunction(list(legs.values())),
                no_causal_mediation_claim=True))
    dual=[]
    for ci,(root,arm) in enumerate(LOGICAL):
        inherited=capability['C_dual_control_conditions'][ci]
        need((inherited['root_slot'],inherited['C_arm'])==(root,arm),'original four C logical slots')
        group=pairs[2*ci:2*ci+2]
        need(conjunction([p['conditions']['immediate_cost'] for p in group]) is inherited['C_immediate_cost_both_controls'],
             'preserve known immediate-cost conjunction including false/null')
        dual.append(dict(root_slot=root,C_arm=arm,
            original_immediate_cost_both_controls=inherited['C_immediate_cost_both_controls'],
            full_chain_both_controls=conjunction([p['full_chain'] for p in group]),
            native_success_U=None if ci else branches[0]['original_native_success']))
    return dict(schema=NAME+'.arithmetic_only',protocol=protocol(),branches=branches,U_Z2_pairs=gains,
        C_control_pairs=pairs,C_dual_control_conditions=dual,
        authentication_performed=False,production_consumer_ready=False,
        new_neural_calls=0,new_physics_calls=0,new_AD_calls=0,**FLAGS)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('describe',))
    ap.parse_args(argv)
    print(json.dumps(protocol(),sort_keys=True,separators=(',',':'),allow_nan=False))


if __name__=='__main__':
    main()
