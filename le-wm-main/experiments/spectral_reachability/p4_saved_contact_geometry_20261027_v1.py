"""Offline saved-state geometry, never force contact, dynamics or admission."""
import argparse
import ast
import json
import sysconfig
from pathlib import Path

import numpy as np
import pymunk

from experiments.spectral_reachability import p4_feedback_admission_20261027_v1 as saved

TAG = 'p4_saved_contact_geometry_20261027_v1'
REPO = Path(__file__).resolve().parents[2]
SITE = Path(sysconfig.get_paths()['purelib'])
ENV = SITE/'stable_worldmodel/envs/pusht/env.py'
# Reviewed installed official source. Different geometry/version must be reviewed anew.
ENV_SHA = 'c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2'
P4 = REPO/'results/p4_feedback_matched_20261027_v2_cuda_main_v1/report.json'
CAP = REPO/'results/pusht_geometry_capability_20261027_v1_cuda_v1/report.json'


def contract(inputs):
    """Freeze geometry BEFORE reading trajectories or outcome-bearing reports.

    Body/Poly are unattached static geometry objects: no Space, env or step.
    Use official transforms, including nonzero COG, rather than assume an origin.
    """
    saved.require(inputs.hash(ENV) == ENV_SHA, 'unsupported official geometry source')
    tree = ast.parse(inputs.raw(ENV))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PushT')
    tee = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'add_tee')
    values = {}
    # Evaluate only a whitelist of literal arithmetic from the reviewed AST.
    def arithmetic(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return n.value
        if isinstance(n, ast.Name) and n.id in values:
            return values[n.id]
        if isinstance(n, (ast.List, ast.Tuple)):
            return [arithmetic(v) for v in n.elts]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -arithmetic(n.operand)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.Div)):
            a, b = arithmetic(n.left), arithmetic(n.right)
            return a*b if isinstance(n.op, ast.Mult) else a/b
        raise ValueError('unsupported polygon expression')
    for node in tee.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('scale', 'length', 'vertices1', 'vertices2'):
                values[name] = arithmetic(node.value)
    a, b = [np.asarray(values[k], dtype=float) for k in ('vertices1', 'vertices2')]
    saved.require(a.tolist() == [[-60,30],[60,30],[60,0],[-60,0]] and
                  b.tolist() == [[-15,30],[-15,120],[15,120],[15,30]], 'unsupported tee union')
    # Exact external boundary of the two source rectangles; exclude internal seam.
    boundary = np.array([a[3], a[2], a[1], b[3], b[2], b[1], b[0], a[0]])
    space_source = inputs.raw(SITE/'pymunk/space.py').decode()
    saved.require('It defaults to 0.1.' in space_source, 'collision-slop contract unavailable')
    for name in ('body.py', 'shapes.py', '__init__.py'):
        inputs.hash(SITE/'pymunk'/name)
    # Source hash above covers default variations: only positions/block angle
    # randomized; agent circle scale40, block T. add_tee ignores scale40 ->30.
    return dict(polygons=[a.tolist(), b.tolist()], boundary=boundary.tolist(),
        pusher_radius=15.0, collision_slop=0.1, source_sha256=ENV_SHA,
        threshold_policy='near_contact: signed boundary clearance <= documented collision_slop 0.1; no fitted buffer',
        movement_policy='report exact deltas; sub_slop flags use same length tolerance, not a noise/force model',
        units='native 512-window environment length units, not resized 224 RGB pixels',
        transform='official pymunk Body.angle then Body.position from saved _get_obs; local_to_world; nonzero COG retained')


def world_boundary(state, geometry):
    state = saved.array(state, (7,)).astype(float)
    body = pymunk.Body(1, 1)
    shapes = [pymunk.Poly(body, v) for v in geometry['polygons']]
    body.center_of_gravity = (shapes[0].center_of_gravity+shapes[1].center_of_gravity)/2
    body.angle = float(state[4])
    body.position = tuple(state[2:4])
    return np.array([body.local_to_world(tuple(v)) for v in geometry['boundary']])


def signed_polygon_distance(point, polygon):
    """Exact Euclidean point-to-boundary distance, negative in concave solid."""
    point = saved.array(point, (2,)).astype(float)
    polygon = np.asarray(polygon, dtype=float)
    saved.require(polygon.shape == (8,2) and np.isfinite(polygon).all(), 'eight-vertex T boundary required')
    ends = np.roll(polygon, -1, axis=0)
    edge = ends-polygon
    denom = np.sum(edge*edge, axis=1)
    saved.require(np.all(denom > 0), 'degenerate polygon')
    t = np.clip(np.sum((point-polygon)*edge, axis=1)/denom, 0, 1)
    nearest = polygon+t[:,None]*edge
    distance = float(np.min(np.linalg.norm(nearest-point, axis=1)))
    inside = False
    for a,b in zip(polygon, ends):
        if (a[1] > point[1]) != (b[1] > point[1]):
            if point[0] < (b[0]-a[0])*(point[1]-a[1])/(b[1]-a[1])+a[0]:
                inside = not inside
    return -distance if inside else distance


def metrics(state, geometry):
    state = saved.array(state, (7,))
    d = signed_polygon_distance(state[:2], world_boundary(state, geometry))
    gap = d-geometry['pusher_radius']
    return dict(agent_center_to_block_signed_distance=d, signed_surface_clearance=gap,
                nonnegative_separation=max(0.0,gap), near_contact=gap <= geometry['collision_slop'],
                geometric_overlap=gap <= 0, force_contact=None, dynamic_contact=None)


def label(state, previous, geometry):
    result = metrics(state, geometry)
    result['state'] = list(state)
    if previous is None:
        result.update(agent_delta=None, agent_displacement=None, block_delta=None,
            block_displacement=None, block_angle_delta=None, block_max_vertex_displacement=None,
            clearance_delta=None, flags=dict(approaching=None, separating=None,
            block_sub_slop_motion=None, near_contact_with_block_motion=None, task_closure=None))
        return result
    previous = saved.array(previous, (7,))
    state = np.asarray(state)
    ad, bd = state[:2]-previous[:2], state[2:4]-previous[2:4]
    angle = float(np.arctan2(np.sin(state[4]-previous[4]),np.cos(state[4]-previous[4])))
    vertex = float(np.max(np.linalg.norm(world_boundary(state,geometry)-world_boundary(previous,geometry),axis=1)))
    dg = result['signed_surface_clearance']-metrics(previous,geometry)['signed_surface_clearance']
    slop = geometry['collision_slop']
    result.update(agent_delta=ad.tolist(), agent_displacement=float(np.linalg.norm(ad)),
        block_delta=bd.tolist(), block_displacement=float(np.linalg.norm(bd)),
        block_angle_delta=angle, block_max_vertex_displacement=vertex, clearance_delta=dg,
        flags=dict(approaching=dg < -slop, separating=dg > slop,
            block_sub_slop_motion=vertex <= slop,
            near_contact_with_block_motion=result['near_contact'] and vertex > slop,
            task_closure=None))
    return result


def old_seals(inputs, report):
    seals = report.get('source_seals', report.get('resources',{}).get('source_seals',[]))
    saved.require(len(seals) == 3, 'three historical source seals required')
    verified = [inputs.seal(s) for s in seals]
    manifests = [inputs.json(Path(s['output_dir'])/'manifest.json') for s in seals]
    saved.require(any(e['source'] == str(ENV) and e['sha256'] == ENV_SHA
        for m in manifests for e in m['files']), 'official geometry not bound to historical run')
    return verified


def p4_rows(inputs, path, geometry):
    report = inputs.json(path)
    verification = old_seals(inputs, report)
    folder = path.parent
    saved.require(len(report['rows']) == 8 and {(r['root_sha256'],r['arm']) for r in report['rows']} ==
        {(r,a) for r in saved.ROOTS for a in saved.ARMS}, 'P4 complete unique census required')
    rows = []
    for row in report['rows']:
        i, arm = row['root_index'], row['arm']
        saved.require(saved.ROOTS[i] == row['root_sha256'], 'P4 root mismatch')
        saved.require(inputs.json(folder/f'root_{i}.{arm}.json') == row, 'P4 branch binding')
        previous = None  # No saved initial physical state; never reconstruct via physics.
        steps = []
        saved.require(len(row['rounds']) == 2, 'P4 two rounds required')
        for k, entry in enumerate(row['rounds']):
            prefix = str(folder/f'root_{i}.{arm}.round_{k}')
            saved.require(inputs.hash(prefix+'.decision.json') == entry['decision_sha256'], 'P4 decision binding')
            decision = inputs.json(prefix+'.decision.json')
            saved.require(decision['root_sha256'] == row['root_sha256'] and decision['arm'] == arm
                and decision['round'] == k, 'P4 decision identity')
            execution = inputs.json(prefix+'.execution.json')
            saved.require(execution == entry['execution'] and execution['completed_commands'] == 5
                and len(execution['trace']) == 5, 'P4 exact five-step receipt required')
            terminal = inputs.npy(prefix+'.terminal.npy', (224,224,3),np.dtype('uint8'))
            saved.require(saved.rgb_sha(terminal) == execution['terminal_rgb_sha256'], 'P4 terminal RGB binding')
            for j, event in enumerate(execution['trace']):
                saved.require(inputs.json(prefix+f'.step_{j+1}.json') == event and event['primitive_step'] == k*5+j+1,
                    'P4 individual state receipt differs')
                saved.require(event['command'] == decision['executed_packet'][j], 'P4 command differs')
                saved.require(len(event['rgb_sha256']) == 64, 'P4 RGB hash missing')
                if j == 4:
                    saved.require(event['rgb_sha256'] == saved.rgb_sha(terminal), 'terminal trace RGB differs')
                item = label(event['state'],previous,geometry)
                item.update(primitive_step=event['primitive_step'],round=k,command=event['command'],
                    rgb=dict(recorded_array_sha256=event['rgb_sha256'],file_verified=j == 4,
                        file=prefix+'.terminal.npy' if j == 4 else None,
                        limitation=None if j == 4 else 'per-step RGB bytes not persisted; hash witness only'))
                steps.append(item)
                previous = event['state']
        rows.append(dict(dataset='P4',root_index=i,root_sha256=row['root_sha256'],arm=arm,
            initial_state_available=False,steps=steps))
    return rows, verification


def capability_rows(inputs, path, geometry):
    report = inputs.json(path)
    verification = old_seals(inputs,report)
    folder = path.parent
    physical = inputs.json(folder/'physical_report.json')
    protocol = inputs.json(folder/'protocol.json')
    saved.require(inputs.hash(folder/'protocol.json') == report['protocol_sha256'], 'capability protocol binding')
    plans = protocol['plans']
    rows = physical['rows']
    saved.require(report['rows'] == rows, 'capability final/physical report binding')
    saved.require(len(rows) == len(plans) == 123 and
        {(r['root_sha256'],r['method']) for r in rows} == {(p['root_sha256'],p['method']) for p in plans}
        and len({r['method'] for r in rows}) == 123, '123 unique capability branches required')
    output = []
    for row in rows:
        plan = next(p for p in plans if p['root_sha256'] == row['root_sha256'] and p['method'] == row['method'])
        saved.require(row['selection_sha256'] == plan['selection_sha256'] and row['root_index'] == plan['root_index'],
            'capability sealed plan identity')
        branch = folder/f'root_{row["root_index"]}_{row["method"]}_replay0.physical.json'
        saved.require(inputs.json(branch) == row, 'capability branch differs')
        journal = Path(row['journal_path'])
        saved.require(inputs.hash(journal) == row['journal_sha256'], 'capability journal binding')
        events = [saved.strict_json(v) for v in inputs.raw(journal).splitlines()]
        initial = events[0]
        saved.require(initial['kind'] == 'initial' and initial['state'] == row['context_identity']['state'], 'initial state differs')
        observations = [e for e in events if e['kind'] == 'observation']
        saved.require(len(observations) == row['completed_commands'] == 10 and
            [e['primitive_step'] for e in observations] == list(range(1,11)), 'capability ten steps required')
        for e,t in zip(observations,row['trace'],strict=True):
            saved.require(all(e[k] == v for k,v in t.items()), 'journal/branch state differs')
            saved.require(e['command'] == plan['commands'][e['primitive_step']-1], 'capability plan command differs')
        previous = initial['state']
        steps = []
        for event in [initial]+observations:
            image = Path(event['rgb_path'])
            saved.require(inputs.hash(image) == event['rgb_file_sha256'], 'capability RGB file binding')
            item = label(event['state'],None if event is initial else previous,geometry)
            item.update(primitive_step=event.get('primitive_step',0),
                command=event.get('command'),rgb=dict(file=str(image),file_verified=True,
                file_sha256=event['rgb_file_sha256']))
            steps.append(item)
            previous = event['state']
        output.append(dict(dataset='capability',root_index=row['root_index'],root_sha256=row['root_sha256'],
            method=row['method'],initial_state_available=True,steps=steps))
    return output,verification


def summarize(rows):
    summaries = []
    for row in rows:
        suffix = [s for s in row['steps'] if s['primitive_step'] > 5]
        summaries.append({k:v for k,v in row.items() if k != 'steps'} | dict(
            suffix_steps=len(suffix),suffix_near_contact_steps=sum(s['near_contact'] for s in suffix),
            suffix_min_clearance=min(s['signed_surface_clearance'] for s in suffix),
            suffix_max_clearance=max(s['signed_surface_clearance'] for s in suffix),
            suffix_max_block_vertex_step=max(s['block_max_vertex_displacement'] for s in suffix),
            suffix_all_sub_slop_block_motion=all(s['flags']['block_sub_slop_motion'] for s in suffix)))
    return summaries


def run(output, p4=P4, capability=CAP):
    output = Path(output).resolve()
    saved.require(output.name.startswith(TAG), 'exclusive basename required')
    output.mkdir(exist_ok=False)
    inputs = saved.Inputs()
    own = Path(__file__).resolve()
    seal = saved.create_seal(files=[own,REPO/f'tests/test_{TAG}.py',REPO/f'docs/{TAG}.md',
        Path(saved.__file__).resolve(),Path(saved.verify_seal.__code__.co_filename),ENV,
        SITE/'pymunk/body.py',SITE/'pymunk/shapes.py',SITE/'pymunk/space.py'],
        args_json=json.dumps(dict(p4=str(p4),capability=str(capability),neural=False,physics=False)),
        output_dir=output/'source_seal')
    try:
        geometry = contract(inputs)
        # Record native extension identity; never construct a Space or simulate.
        import pymunk._chipmunk as chipmunk
        inputs.hash(chipmunk.__file__)
        a, av = p4_rows(inputs,Path(p4),geometry)
        b, bv = capability_rows(inputs,Path(capability),geometry)
        saved.require(sum(len(r['steps']) for r in a) == 80 and sum(len(r['steps'])-1 for r in b) == 1230, 'step census differs')
        rows = a+b
        result = dict(schema=TAG,status='saved_geometry_labeled',geometry=geometry,rows=rows,
            summaries=summarize(rows),historical_source_verification=av+bv,
            census=dict(P4_steps=80,capability_branches=123,capability_steps=1230,
                P4_missing_initial_displacements=8,P4_RGB_files_verified=16,P4_RGB_hash_only_steps=64),
            input_sha256=inputs.hashes,all_inputs_unchanged=inputs.stable(),
            limitations=['snapshot geometry cannot exclude intervening substep contacts',
                'near-contact is not force/dynamic contact or causal pushing',
                'closure remains unknown: no task-outcome thresholds or goal-based fitting',
                'RGB is provenance only; no image-derived segmentation/shape verification'])
    except (OSError,ValueError,TypeError,KeyError,IndexError,StopIteration) as exc:
        result = dict(schema=TAG,status='integrity_fail',error=type(exc).__name__+': '+str(exc))
    result.update(P1_admission=False,P4_admission=False,new_neural_calls=0,new_physics_steps=0,
        audit_source_seal=seal,audit_source_verification=saved.verify_seal(
            output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256']))
    saved.write(output/'report.json',result)
    print(json.dumps(dict(status=result['status'],output=str(output),error=result.get('error')),sort_keys=True))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if run(**vars(args))['status'] == 'saved_geometry_labeled' else 2)
