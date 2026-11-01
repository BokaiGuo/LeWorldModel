"""Offline rigid-motion/goal alignment of six saved feedback suffix traces.

No simulator, neural execution, force inference, fitted threshold or admission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pymunk

from experiments.spectral_reachability import p4_feedback_contact_recovery_20261027_v1 as recovery

geometry, saved = recovery.geometry, recovery.saved
require = saved.require
TAG = 'p4_feedback_motion_alignment_20261027_v1'
REPO = Path(__file__).resolve().parents[2]
DEFAULT_RECOVERY = REPO/'results/p4_feedback_contact_recovery_20261027_v1_main_v1/report.json'
NUMERICAL_ATOL = 1e-9  # Reconstruction only; never a physical/goal/contact cutoff.


def wrap(angle):
    return float(np.arctan2(np.sin(angle),np.cos(angle)))


def direction(displacement, goal_vector):
    """Finite displacement alignment and exact fixed-target distance change.

    Projection is a tangent-direction diagnostic, not guaranteed improvement
    when overshooting; exact error reduction is independently reported.
    """
    d, e = np.asarray(displacement,dtype=float), np.asarray(goal_vector,dtype=float)
    require(d.ndim == e.ndim == 1 and d.shape == e.shape and d.size > 0 and
        np.isfinite(d).all() and np.isfinite(e).all(), 'aligned finite vectors required')
    dn, en = float(np.linalg.norm(d)), float(np.linalg.norm(e))
    dot = float(d@e)
    after = float(np.linalg.norm(e-d))
    projection = None if en == 0 else dot/en
    return dict(displacement_norm=dn,goal_error_before=en,goal_error_after=after,
        exact_goal_distance_reduction=en-after,dot_with_goal_vector=dot,
        signed_goal_axis_displacement=projection,
        perpendicular_displacement=None if projection is None else float(np.linalg.norm(d-e*(dot/(en*en)))),
        cosine=None if dn == 0 or en == 0 else float(np.clip(dot/(dn*en),-1,1)))


def alignment(displacement, reference):
    """Vector comparison only, not hypothetical task progress of the pusher."""
    r = direction(displacement,reference)
    return dict(source_displacement_norm=r['displacement_norm'],reference_vector_norm=r['goal_error_before'],
        dot=r['dot_with_goal_vector'],signed_reference_axis_displacement=r['signed_goal_axis_displacement'],
        perpendicular_displacement=r['perpendicular_displacement'],cosine=r['cosine'])


def body_frame(state, g):
    """Official snapshot origin/COG convention, no Space or state stepping."""
    state = saved.array(state,(7,)).astype(float)
    body = pymunk.Body(1,1)
    parts = [pymunk.Poly(body,v) for v in g['polygons']]
    cog_local = (parts[0].center_of_gravity+parts[1].center_of_gravity)/2
    body.center_of_gravity = cog_local
    body.angle = float(state[4]); body.position = tuple(state[2:4])
    local = np.asarray(g['boundary'],dtype=float)
    vertices = np.array([body.local_to_world(tuple(v)) for v in local])
    np.testing.assert_allclose(vertices,geometry.world_boundary(state,g),rtol=0,atol=NUMERICAL_ATOL)
    origin = np.array(body.local_to_world((0,0)))
    require(np.allclose(origin,state[2:4],rtol=0,atol=NUMERICAL_ATOL), 'saved position is not supported body origin')
    c,s = np.cos(state[4]),np.sin(state[4])
    return dict(body=body,vertices=vertices,local=local,origin=origin,
        cog=np.array(body.local_to_world(tuple(cog_local))),cog_local=np.array(cog_local),
        rotation=np.array([[c,-s],[s,c]]),state=state)


def frame_receipt(f):
    return dict(body_origin_world=f['origin'].tolist(),cog_local=f['cog_local'].tolist(),
        cog_world=f['cog'].tolist(),boundary_world=f['vertices'].tolist(),
        pusher_body_xy=list(f['body'].world_to_local(tuple(f['state'][:2]))))


def anchor(before, after, goal, reference):
    """Nearest solid-boundary anchor at one endpoint, tracked as material point.

    A nearest geometric boundary point/normal is NOT a measured contact point,
    pressure center, force direction, slip constraint or instantaneous velocity.
    """
    point, vertices = reference['state'][:2],reference['vertices']
    ends = np.roll(vertices,-1,axis=0)
    edges = ends-vertices
    den = np.sum(edges*edges,axis=1)
    require(np.all(den > 0), 'degenerate boundary edge')
    t = np.clip(np.sum((point-vertices)*edges,axis=1)/den,0,1)
    nearest = vertices+t[:,None]*edges
    distances = np.linalg.norm(point-nearest,axis=1)
    index = int(np.argmin(distances))
    tied = np.flatnonzero(np.abs(distances-distances[index]) <= NUMERICAL_ATOL).tolist()
    local = reference['local'][index]+t[index]*(reference['local'][(index+1)%8]-reference['local'][index])
    material = [np.array(f['body'].local_to_world(tuple(local))) for f in (before,after,goal)]
    normal = np.array([edges[index,1],-edges[index,0]])/np.sqrt(den[index])
    tangent = edges[index]/np.sqrt(den[index])
    area2 = float(np.sum(vertices[:,0]*ends[:,1]-ends[:,0]*vertices[:,1]))
    require(area2 > 0, 'unsupported boundary winding for outward normal')
    require(np.allclose(nearest[index],reference['body'].local_to_world(tuple(local)),rtol=0,atol=NUMERICAL_ATOL),
        'material anchor transform differs')
    pusher_delta = after['state'][:2]-before['state'][:2]
    material_delta = material[1]-material[0]
    relative = pusher_delta-material_delta
    return dict(edge_index=index,edge_fraction=float(t[index]),tied_edge_indices=tied,
        normal_nonunique_at_corner=len(tied)>1,material_point_body=local.tolist(),
        nearest_boundary_point_world=nearest[index].tolist(),
        agent_center_to_boundary_distance=float(distances[index]),
        outward_unit_normal_world=normal.tolist(),tangent_unit_world=tangent.tolist(),
        material_point_before_world=material[0].tolist(),material_point_after_world=material[1].tolist(),
        material_point_goal_world=material[2].tolist(),material_displacement_world=material_delta.tolist(),
        pusher_minus_material_displacement_world=relative.tolist(),
        pusher_inward_displacement=float(pusher_delta@(-normal)),
        pusher_tangential_displacement=float(pusher_delta@tangent),
        material_inward_displacement=float(material_delta@(-normal)),
        pusher_relative_inward_displacement=float(relative@(-normal)),
        pusher_vs_material_motion=alignment(pusher_delta,material_delta),
        pusher_vs_material_goal_direction=alignment(pusher_delta,material[2]-material[0]),
        material_vs_goal=direction(material_delta,material[2]-material[0]),
        force_direction=None,contact_point_measured=False)


def motion(before_state, after_state, goal_state, g):
    before,after,goal = [body_frame(s,g) for s in (before_state,after_state,goal_state)]
    d = after['origin']-before['origin']
    dc = after['cog']-before['cog']
    da = after['state'][:2]-before['state'][:2]
    dv = after['vertices']-before['vertices']
    rotation = after['rotation']@before['rotation'].T
    rotational = (before['vertices']-before['origin'])@(rotation-np.eye(2)).T
    rotational_cog = (before['vertices']-before['cog'])@(rotation-np.eye(2)).T
    origin_error = float(np.max(np.abs(dv-(d+rotational))))
    cog_error = float(np.max(np.abs(dv-(dc+rotational_cog))))
    require(max(origin_error,cog_error) <= NUMERICAL_ATOL, 'rigid translation/rotation reconstruction failed')
    angle = wrap(after['state'][4]-before['state'][4])
    desired_before = wrap(goal['state'][4]-before['state'][4])
    desired_after = wrap(goal['state'][4]-after['state'][4])
    gb,ga = geometry.metrics(before_state,g),geometry.metrics(after_state,g)
    vertex_direction = direction(dv.ravel(),(goal['vertices']-before['vertices']).ravel())
    vb = float(np.sqrt(np.mean(np.sum((goal['vertices']-before['vertices'])**2,axis=1))))
    va = float(np.sqrt(np.mean(np.sum((goal['vertices']-after['vertices'])**2,axis=1))))
    pose_before,pose_after = recovery.pose(before_state,goal_state),recovery.pose(after_state,goal_state)
    return dict(before=frame_receipt(before),after=frame_receipt(after),
        object_origin_displacement_world=d.tolist(),object_cog_displacement_world=dc.tolist(),
        pusher_displacement_world=da.tolist(),
        object_origin_displacement_in_before_body_axes=(before['rotation'].T@d).tolist(),
        pusher_displacement_in_before_body_axes=(before['rotation'].T@da).tolist(),
        goal_origin_vector_in_before_body_axes=(before['rotation'].T@(goal['origin']-before['origin'])).tolist(),
        origin_vs_fixed_goal=direction(d,goal['origin']-before['origin']),
        cog_vs_fixed_goal=direction(dc,goal['cog']-before['cog']),
        pusher_vs_object_origin_motion=alignment(da,d),pusher_vs_object_cog_motion=alignment(da,dc),
        pusher_vs_fixed_goal_origin_direction=alignment(da,goal['origin']-before['origin']),
        angular=dict(principal_displacement=angle,desired_shortest_before=desired_before,
            desired_shortest_after=desired_after,absolute_error_reduction=abs(desired_before)-abs(desired_after),
            signed_advance_along_previous_goal_direction=None if desired_before == 0 else float(np.sign(desired_before)*angle),
            half_turn_goal_direction_ambiguous=abs(abs(desired_before)-np.pi)<=NUMERICAL_ATOL),
        corresponding_vertices=dict(displacements_world=dv.tolist(),goal_alignment=vertex_direction,
            goal_rms_distance_before=vb,goal_rms_distance_after=va,exact_goal_rms_reduction=vb-va,
            rotation_component_about_origin_world=rotational.tolist(),
            rotation_component_about_cog_world=rotational_cog.tolist(),
            reconstruction_max_abs_error_origin=origin_error,reconstruction_max_abs_error_cog=cog_error),
        contact_geometry=dict(before_near=gb['near_contact'],after_near=ga['near_contact'],
            before_clearance=gb['signed_surface_clearance'],after_clearance=ga['signed_surface_clearance'],
            either_endpoint_near=gb['near_contact'] or ga['near_contact'],
            before_anchor=anchor(before,after,goal,before),after_anchor=anchor(before,after,goal,after),
            actual_contact_during_interval=None,force_causality=None),
        original_pose=dict(before=pose_before,after=pose_after,
            cost_change=pose_after['pose_cost']-pose_before['pose_cost'],
            translation_cost_change=pose_after['xy_distance']-pose_before['xy_distance'],
            angle_cost_change=20*(pose_after['angle_distance']-pose_before['angle_distance'])))


def analyze_steps(steps, goal, g):
    require(len(steps) == 11 and [s['primitive_step'] for s in steps] == list(range(11)),
        'complete exact snapshots 0..10 required; no shortened terminal')
    records = []
    for s in steps:
        metrics = geometry.metrics(s['state'],g)
        require(s['near_contact'] is metrics['near_contact'] and
            abs(s['signed_surface_clearance']-metrics['signed_surface_clearance']) <= NUMERICAL_ATOL,
            'saved geometry/threshold changed')
    for k in range(1,11):
        value = motion(steps[k-1]['state'],steps[k]['state'],goal,g)
        value.update(primitive_step=k,phase='prefix' if k<=5 else 'suffix',command=steps[k]['command'])
        records.append(value)
    suffix = records[5:]
    near = [v for v in suffix if v['contact_geometry']['either_endpoint_near']]
    cosines = [v['pusher_vs_object_origin_motion']['cosine'] for v in near
               if v['pusher_vs_object_origin_motion']['cosine'] is not None]
    net = motion(steps[5]['state'],steps[10]['state'],goal,g)
    for summed,wanted in (
        (sum(v['origin_vs_fixed_goal']['exact_goal_distance_reduction'] for v in suffix),net['origin_vs_fixed_goal']['exact_goal_distance_reduction']),
        (sum(v['angular']['absolute_error_reduction'] for v in suffix),net['angular']['absolute_error_reduction']),
        (sum(v['corresponding_vertices']['exact_goal_rms_reduction'] for v in suffix),net['corresponding_vertices']['exact_goal_rms_reduction'])):
        require(abs(summed-wanted) <= NUMERICAL_ATOL, 'fixed-goal progress does not telescope')
    return dict(intervals=records,prefix_net=motion(steps[0]['state'],steps[5]['state'],goal,g),suffix_net=net,
        summary=dict(origin_goal_distance_reduction=net['origin_vs_fixed_goal']['exact_goal_distance_reduction'],
            cog_goal_distance_reduction=net['cog_vs_fixed_goal']['exact_goal_distance_reduction'],
            absolute_angle_error_reduction=net['angular']['absolute_error_reduction'],
            corresponding_vertex_goal_rms_reduction=net['corresponding_vertices']['exact_goal_rms_reduction'],
            original_pose_cost_change=net['original_pose']['cost_change'],
            suffix_origin_path_length=sum(v['origin_vs_fixed_goal']['displacement_norm'] for v in suffix),
            suffix_absolute_principal_rotation=sum(abs(v['angular']['principal_displacement']) for v in suffix),
            suffix_signed_principal_rotation_sum=sum(v['angular']['principal_displacement'] for v in suffix),
            suffix_either_endpoint_near_intervals=len(near),
            near_interval_origin_goal_distance_reduction_sum=sum(v['origin_vs_fixed_goal']['exact_goal_distance_reduction'] for v in near),
            near_interval_pusher_vs_origin_cosine_defined_count=len(cosines),
            near_interval_pusher_vs_origin_cosine_mean=None if not cosines else float(np.mean(cosines)),
            no_force_or_dynamic_contact_inference=True))


def audit(recovery_report):
    inputs = saved.Inputs()
    g = geometry.contract(inputs)  # Freeze before any new trajectory/goal inspection.
    prior = inputs.json(recovery_report)
    require(prior['schema'] == recovery.TAG and prior['status'] == 'saved_contact_recovery_recorded',
        'completed saved recovery report required')
    inputs.seal(prior['audit_source_seal'])
    args = inputs.json(Path(prior['audit_source_seal']['output_dir'])/'args.json')
    physical_path = Path(args['report'])
    fresh = recovery.audit(physical_path)  # Existing read-only arithmetic, never runtime run/prepare.
    for key in ('schema','status','geometry','rows','paired_roles','pose_contract','compiler_report_sha256'):
        require(prior[key] == fresh[key], 'saved recovery re-audit differs: '+key)
    require(prior['P1_admission'] is prior['P4_admission'] is False and prior['geometry'] == g,
        'geometry/admission boundary changed')
    for report in (prior,fresh):
        for path, expected in report['input_sha256'].items():
            require(inputs.hash(path) == expected, 'recovery bound input changed: '+path)
    physical_report = inputs.json(physical_path)
    expected = {(role,arm) for role in recovery.physical.ROLES for arm in recovery.physical.ARMS}
    rows = prior['rows']
    require(len(rows) == 6 and {(r['role'],r['arm']) for r in rows} == expected, 'six unique slots required')
    goals = [r['context_identity']['goal'] for r in physical_report['rows']]
    require(len(goals) == 6 and all(goal == goals[0] for goal in goals), 'one fixed original physical goal required')
    goal = saved.array(goals[0],(7,)).tolist()
    outputs = []
    for row in rows:
        require(row['physical_complete'], 'complete saved six-branch panel required')
        value = analyze_steps(row['steps'],goal,g)
        for k,v in row['steps'][10]['pose'].items():
            recovery.close(value['suffix_net']['original_pose']['after'][k],v,'fixed goal/original pose differs')
        value.update(method=row['method'],root_sha256=row['root_sha256'],role=row['role'],arm=row['arm'],
            selected_candidate_id=row['selected_candidate_id'],fixed_goal_state=goal,
            saved_neural_arithmetic=row['saved_neural_arithmetic'])
        outputs.append(value)
    pairs = []
    for role in recovery.physical.ROLES:
        imagined,actual = [next(row for row in outputs if row['role']==role and row['arm']==arm)
                          for arm in recovery.physical.ARMS]
        fields = ('origin_goal_distance_reduction','cog_goal_distance_reduction','absolute_angle_error_reduction',
                  'corresponding_vertex_goal_rms_reduction','original_pose_cost_change')
        pairs.append(dict(role=role,actual_minus_imagined_suffix_summary={
            key:actual['summary'][key]-imagined['summary'][key] for key in fields},
            interpretation='descriptive known-root pair, not an independent effect estimate'))
    return dict(schema=TAG,status='saved_motion_alignment_recorded',geometry=g,
        fixed_goal=frame_receipt(body_frame(goal,g)),fixed_goal_state=goal,rows=outputs,paired_roles=pairs,
        census=dict(branches=6,snapshot_receipts=66,primitive_intervals=60,suffix_intervals=30),
        conventions=dict(world_axes='saved native x/y, no display y-flip; do not infer physical clockwise labels',
            body_origin='saved block.position; local_to_world(0,0)',
            official_cog='mean of the two official Poly centroids; not the body origin or assumed area centroid',
            goal='single fixed original physical state, same local shape/material-vertex correspondence',
            angular='principal atan2(sin,cos); multiple revolutions between snapshots unobservable',
            geometric_anchor='nearest boundary point at either saved endpoint, same local material point tracked through poses',
            numerical_atol=NUMERICAL_ATOL,physical_thresholds='unchanged old collision_slop .1 only for endpoint near labels'),
        input_sha256=inputs.hashes,all_inputs_unchanged=inputs.stable(),
        limitations=['endpoint geometry does not measure actual contact, pressure, force, slip or causal pushing',
            'finite displacements are not instantaneous contact velocities; no substep interpolation',
            'projection/cosine toward a goal is not exact goal-error reduction; overshoot is retained',
            'corresponding-vertex RMS is an explicit diagnostic, not official pose or a new success threshold',
            'one reused known root; no causal, independent efficacy or P1/P4 admission claim'])


def markdown(result):
    if result['status'] != 'saved_motion_alignment_recorded':
        return '# Saved motion alignment\n\nIntegrity failed: '+result['error']+'\n'
    lines = ['# Saved motion alignment','',
        'Positive distance/angle reduction is toward the fixed goal; positive original pose change is worse.',
        'Near labels are endpoint geometry only, not measured force contact. P1/P4 admission=false.','',
        '| Branch | Origin distance reduction | Angle error reduction (rad) | Original pose change |',
        '| --- | ---: | ---: | ---: |']
    for row in result['rows']:
        s = row['summary']
        lines.append(f'| {row["method"]} | {s["origin_goal_distance_reduction"]:.9f} | '
            f'{s["absolute_angle_error_reduction"]:.9f} | {s["original_pose_cost_change"]:.9f} |')
    return '\n'.join(lines)+'\n'


def run(recovery_report, output):
    output = Path(output).resolve()
    require(output.name.startswith(TAG), 'exclusive basename required')
    output.mkdir(exist_ok=False)
    own = Path(__file__).resolve()
    files = [own,REPO/f'tests/test_{TAG}.py',REPO/f'innerdocs/{TAG}.md',Path(recovery.__file__),
        Path(geometry.__file__),Path(saved.__file__),Path(saved.verify_seal.__code__.co_filename),
        geometry.ENV,geometry.SITE/'pymunk/body.py',geometry.SITE/'pymunk/shapes.py',geometry.SITE/'pymunk/space.py']
    seal = saved.create_seal(files=files,args_json=json.dumps(dict(recovery_report=str(Path(recovery_report).resolve()),
        neural=False,physics=False)),output_dir=output/'source_seal')
    try:
        result = audit(recovery_report)
    except (OSError,ValueError,TypeError,KeyError,IndexError,StopIteration,AssertionError) as exc:
        result = dict(schema=TAG,status='integrity_fail',error=type(exc).__name__+': '+str(exc))
    result.update(P1_admission=False,P4_admission=False,new_neural_calls=0,new_physics_steps=0,
        audit_source_seal=seal,audit_source_verification=saved.verify_seal(
            output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256']))
    saved.write(output/'report.json',result)
    with (output/'report.md').open('x') as stream:
        import os
        stream.write(markdown(result)); stream.flush(); os.fsync(stream.fileno())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-report',type=Path,default=DEFAULT_RECOVERY)
    parser.add_argument('--output',type=Path,required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({k:result[k] for k in ('schema','status')} | dict(error=result.get('error'))))
    raise SystemExit(0 if result['status'] == 'saved_motion_alignment_recorded' else 2)
