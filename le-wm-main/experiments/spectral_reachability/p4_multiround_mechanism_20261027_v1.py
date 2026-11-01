"""Saved three-round direction/drift/physical-progress decomposition, not causality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p4_multiround_outcomes_20261027_v1 as outcomes
from experiments.spectral_reachability import p4_feedback_motion_alignment_20261027_v1 as motion

saved, geometry = outcomes.saved, motion.geometry
require = saved.require
TAG = 'p4_multiround_mechanism_20261027_v1'
REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTCOMES = REPO/'results/p4_multiround_outcomes_20261027_v1_main_v1/report.json'


def latent_cost(value,goal,q):
    v,g,m = saved.array(value,(192,)),saved.array(goal,(192,)),saved.array(q,(192,192))
    return float(np.mean((m@(g-v))**2))


def drift(observed,predicted,goal,q):
    """Exact original-Q residual identity; no newly fitted metric/threshold."""
    observed,predicted,goal = [saved.array(v,(192,)) for v in (observed,predicted,goal)]
    q=saved.array(q,(192,192)); d=q@(observed-predicted); r=q@(goal-predicted)
    square=float(np.mean(d*d)); cross=float(-2*np.mean(r*d))
    actual_change=latent_cost(observed,goal,q)-latent_cost(predicted,goal,q)
    require(abs(square+cross-actual_change)<=1e-10, 'feedback residual identity differs')
    return dict(raw_l2=float(np.linalg.norm(observed-predicted)),projected_l2=float(np.linalg.norm(d)),
        projected_mse=square,goal_cross_term=cross,observed_minus_predicted_Q_cost=actual_change,
        drift_vs_predicted_goal_direction=motion.alignment(d,r),identity_error=square+cross-actual_change)


def official_components(state,goal):
    d=saved.array(state,(7,)).astype(float)-saved.array(goal,(7,)).astype(float)
    parts=dict(agent_xy_squared=float(d[:2]@d[:2]),object_xy_squared=float(d[2:4]@d[2:4]),
        unwrapped_angle_squared=float(d[4]**2),pusher_velocity_squared=float(d[5:]@d[5:]))
    require(np.isclose(sum(parts.values()),float(d@d),rtol=0,atol=1e-9), 'official distance decomposition differs')
    return parts


def physical_segment(steps,start,end,goal,g):
    """Actual saved states only; retain absent/partial segment slots, never pad."""
    by={v['primitive_step']:v for v in steps}
    require(len(by)==len(steps),'duplicate physical snapshots')
    if start not in by:
        return dict(status='not_observed',start=start,planned_end=end,last_observed=None,net=None,intervals=[])
    last=start
    while last+1 in by and last<end:
        last+=1
    records=[]
    for step in range(start+1,last+1):
        v=motion.motion(by[step-1]['state'],by[step]['state'],goal,g)
        records.append(dict(primitive_step=step,
            origin_goal_distance_reduction=v['origin_vs_fixed_goal']['exact_goal_distance_reduction'],
            origin_displacement_world=v['object_origin_displacement_world'],
            pusher_displacement_world=v['pusher_displacement_world'],
            principal_rotation=v['angular']['principal_displacement'],
            angle_error_reduction=v['angular']['absolute_error_reduction'],
            pose_change=v['original_pose']['cost_change'],
            before_clearance=v['contact_geometry']['before_clearance'],
            after_clearance=v['contact_geometry']['after_clearance'],
            either_endpoint_geometrically_near=v['contact_geometry']['either_endpoint_near']))
    net=motion.motion(by[start]['state'],by[last]['state'],goal,g)
    for key,expected in (
        ('origin_goal_distance_reduction',net['origin_vs_fixed_goal']['exact_goal_distance_reduction']),
        ('angle_error_reduction',net['angular']['absolute_error_reduction']),
        ('pose_change',net['original_pose']['cost_change'])):
        require(abs(sum(v[key] for v in records)-expected)<=1e-9,'per-step progress does not telescope')
    before=official_components(by[start]['state'],goal);after=official_components(by[last]['state'],goal)
    component_change={k:after[k]-before[k] for k in before}
    net_compact={k:net[k] for k in ('object_origin_displacement_world','object_cog_displacement_world',
        'pusher_displacement_world','object_origin_displacement_in_before_body_axes',
        'pusher_displacement_in_before_body_axes','goal_origin_vector_in_before_body_axes',
        'origin_vs_fixed_goal','cog_vs_fixed_goal','angular','original_pose','pusher_vs_object_origin_motion')}
    return dict(status='complete_segment' if last==end else 'partial_observed_segment',start=start,planned_end=end,
        last_observed=last,net=net_compact,intervals=records,
        official_squared_distance_component_change=component_change,
        official_squared_distance_change=sum(component_change.values()),
        geometric_near_endpoint_intervals=sum(v['either_endpoint_geometrically_near'] for v in records),
        force_contact=None,causal_attribution=None)


def analyze_arm(p,row,events,g):
    arm=row['arm'];goal,q=p['goal'],p['q']
    contexts={e['boundary']:e for e in events if e['kind']=='planning_context' and e.get('arm')==arm}
    selections={e['boundary']:e for e in events if e['kind']=='selection' and e.get('arm')==arm}
    rounds=[]
    for start in (5,10,15):
        physical=physical_segment(row.get('steps',[]),start,start+5,p['physical_goal'],g)
        value=dict(arm=arm,start=start,planned_end=start+5,physical=physical,model=None,
            boundary_feedback=None,next_boundary_prediction_discrepancy=None,
            next_latent_status='not_saved_at20' if start==15 else 'next_boundary_not_observed')
        if start in contexts:
            c=contexts[start]
            value['boundary_feedback']=drift(c['observed_context']['latent_history'][-1],
                c['predicted_context']['latent_history'][-1],goal,q)
            value['boundary_feedback']['selected_context']='observed' if arm=='actual' else 'predicted'
        if start not in selections:
            rounds.append(value)
            continue
        s=selections[start];plan=s['compiler_plan'];chosen=plan['context']['latent_history'][-1]
        winner=next(c for c in plan['candidates'] if c['candidate_id']==plan['selected_candidate_id'])
        pred=winner['predicted_terminal'];before=latent_cost(chosen,goal,q)
        nominal=latent_cost(plan['nominal_terminal'],goal,q);after=latent_cost(pred,goal,q)
        require(abs(after-winner['nonlinear_terminal_cost'])<=1e-12,'saved model score differs')
        value['model']=dict(selected_candidate_id=winner['candidate_id'],selection_sha256=s['selection_sha256'],
            chosen_context_Q_cost=before,zero_action_nominal_Q_cost=nominal,selected_predicted_Q_cost=after,
            predicted_reduction_from_chosen=before-after,predicted_reduction_from_zero_nominal=nominal-after,
            predicted_motion_vs_model_goal=motion.direction(np.asarray(q)@(np.asarray(pred)-np.asarray(chosen)),
                np.asarray(q)@(np.asarray(goal)-np.asarray(chosen))))
        if start+5 in contexts:
            next_observed=contexts[start+5]['observed_context']['latent_history'][-1]
            value['next_boundary_prediction_discrepancy']=drift(next_observed,pred,goal,q)
            value['next_boundary_prediction_discrepancy']['conditioning_scope']=(
                'one-packet prediction from observed start' if arm=='actual' else 'rolling imagined-start prediction; includes accumulated context drift')
            observed_start=contexts[start]['observed_context']['latent_history'][-1]
            value['observed_latent_Q_cost_change']=latent_cost(next_observed,goal,q)-latent_cost(observed_start,goal,q)
            value['observed_latent_motion_vs_goal']=motion.direction(
                np.asarray(q)@(np.asarray(next_observed)-np.asarray(observed_start)),
                np.asarray(q)@(np.asarray(goal)-np.asarray(observed_start)))
            value['next_latent_status']='saved_next_boundary'
        rounds.append(value)
    return dict(arm=arm,classification=row['classification'],rounds=rounds,
        complete20=row['classification']=='complete20',
        observed_pose_change_sum=(sum(v['physical']['net']['original_pose']['cost_change']
            for v in rounds if v['physical']['net'] is not None)
            if any(v['physical']['net'] is not None for v in rounds) else None),
        missing_terminal20_latent=True,force_or_causal_claim=False)


def audit(outcomes_report):
    inputs=saved.Inputs()
    g=geometry.contract(inputs)  # Original T/COG/slop contract before new outcomes.
    prior=inputs.json(REPO/'results/p4_saved_contact_geometry_20261027_v1_main_v1/report.json')
    inputs.seal(prior['audit_source_seal'])
    require(prior['geometry']==g,'fixed original geometry contract changed')
    import pymunk._chipmunk as chipmunk
    inputs.hash(chipmunk.__file__)
    path=Path(outcomes_report).resolve();old=inputs.json(path)
    require(old['schema']==outcomes.TAG and old['status']=='saved_arithmetic_audit_passed','validated outcomes report required')
    inputs.seal(old['audit_source_seal'])
    for f,h in old['inputs'].items():
        require(inputs.hash(f)==h,'original outcomes input changed: '+f)
    reports=[Path(f) for f in old['inputs'] if Path(f).name=='report.json' and Path(f).parent.name.startswith(outcomes.runtime.TAG)]
    require(len(reports)==1,'unique original runtime report binding required')
    runtime_path=reports[0]
    recomputed=outcomes.audit(runtime_path,inputs=inputs)  # Pure saved algebra, never run/prepare.
    for key in ('rows','comparison','saved_cost','metric_contract','runtime_status','root_sha256','physical_goal'):
        require(recomputed[key]==old[key],'saved outcomes arithmetic does not reproduce: '+key)
    p=inputs.json(runtime_path.parent/'protocol.json')
    events=[saved.strict_json(line) for line in inputs.raw(runtime_path.parent/'runtime.events.jsonl',outcomes.MAX_JOURNAL_BYTES).splitlines()]
    rows=[analyze_arm(p,row,events,g) for row in recomputed['rows']]
    for row,original in zip(rows,recomputed['rows']):
        if row['complete20']:
            require(abs(row['observed_pose_change_sum']-(original['exact20']['pose']['pose_cost']-
                original['prefix']['pose']['pose_cost']))<=1e-9,'three-round pose changes do not telescope')
    inputs.stable()
    return dict(schema=TAG,status='saved_round_decomposition_recorded',rows=rows,geometry=g,
        outcomes_report=str(path),original_comparison=recomputed['comparison'],inputs=inputs.hashes,
        inputs_unchanged=True,new_neural_calls=0,new_physics_steps=0,
        P1_admission=False,P4_admission=False,causal_attribution=False,
        limitations=['one reused calibration root; no independent mechanism or efficacy test',
            'step20 has physical state/RGB but no saved neural encoding; terminal latent diagnostics unavailable',
            'predicted latent-Q progress and physical pose progress have different units; no fitted mapping',
            'snapshot geometric near-contact is not force or substep dynamic contact'])


def run(output,outcomes_report=DEFAULT_OUTCOMES):
    output=Path(output).resolve()
    require(output.name.startswith(TAG) and not output.exists(),'new exclusive mechanism output required')
    output.mkdir(exist_ok=False)
    files=[Path(__file__),REPO/f'tests/test_{TAG}.py',REPO/f'innerdocs/{TAG}.md',
        Path(outcomes.__file__),Path(motion.__file__),Path(geometry.__file__),Path(saved.__file__),
        geometry.ENV,geometry.SITE/'pymunk/body.py',geometry.SITE/'pymunk/shapes.py',geometry.SITE/'pymunk/space.py']
    seal=saved.create_seal(files=files,args_json=json.dumps(dict(outcomes_report=str(Path(outcomes_report).resolve()),
        new_neural_calls=0,new_physics_steps=0)),output_dir=output/'source_seal')
    try:
        result=audit(outcomes_report)
        result.update(audit_source_seal=seal,source_verification=saved.verify_seal(
            output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256']))
        saved.write(output/'report.json',result)
        print(json.dumps(dict(status=result['status'],rounds=sum(len(r['rounds']) for r in result['rows']))))
        return result
    except (Exception,KeyboardInterrupt) as exc:
        saved.write(output/'failure.json',dict(status='audit_fail_closed',type=type(exc).__name__,message=str(exc),audit_source_seal=seal))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outcomes-report',type=Path,default=DEFAULT_OUTCOMES)
    p.add_argument('--output',type=Path,required=True)
    run(**vars(p.parse_args()))
