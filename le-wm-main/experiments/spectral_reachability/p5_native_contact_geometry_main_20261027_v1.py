"""Saved native geometry/PD arithmetic only, no environment or model imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

NAME = 'p5_native_contact_geometry_main_20261027_v1'
ARMS = ('compiled', 'nominal', 'factual_reference')
REPORT_SHA = 'e2f8245f8f69e41eeb35ec5507d8e744f7f32e1e8f76948756927ddf42c44ae6'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    p = Path(path).absolute()
    require(p.is_file() and not any(v.is_symlink() for v in (p, *p.parents)), 'regular nonsymlink input')
    return hashlib.sha256(p.read_bytes()).hexdigest()


def world_vertices(vertices, body):
    angle = body['angle']
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    # body.position is the world position of its LOCAL ORIGIN, not the CoG.
    return np.asarray(vertices, float) @ rotation.T + np.asarray(body['position'], float)


def polygon_gap(point, polygon, radius):
    p, v = np.asarray(point, float), np.asarray(polygon, float)
    require(p.shape == (2,) and v.ndim == 2 and v.shape[1] == 2 and len(v) >= 3 and
            np.isfinite(p).all() and np.isfinite(v).all() and radius > 0, 'finite convex geometry')
    edge = np.roll(v, -1, axis=0)-v
    squared = np.sum(edge*edge, axis=1)
    require(np.all(squared > 0), 'nondegenerate edges')
    cross = edge[:, 0]*(p-v)[:, 1]-edge[:, 1]*(p-v)[:, 0]
    # In this audit every circle CENTER is outside both T rectangles. Do not
    # silently reuse per-piece signed distances for interior union geometry.
    inside = np.all(cross >= 0) or np.all(cross <= 0)
    require(not inside, 'outside-center audit cannot handle polygon-interior center')
    t = np.clip(np.sum((p-v)*edge, axis=1)/squared, 0, 1)
    return float(np.min(np.linalg.norm(v+t[:, None]*edge-p, axis=1)) - radius)


def geometry(public):
    circles = [s for s in public['shapes'] if s['body'] == 'agent']
    polys = [s for s in public['shapes'] if s['body'] == 'block']
    require(len(circles) == 1 and circles[0]['type'] == 'Circle' and circles[0]['radius'] == 15 and
            circles[0]['offset'] == [0., 0.] and len(polys) == 2 and
            all(s['type'] == 'Poly' and s['radius'] == 0 for s in polys), 'fixed native T geometry')
    for s in circles + polys:
        require(not s['sensor'] and s['collision_type'] == 0 and s['filter'] ==
                dict(categories=4294967295, group=0, mask=4294967295), 'unaltered collision support')
    return [world_vertices(s['vertices'], public['bodies']['block']) for s in polys], 15.


def gap(point, polygons, radius):
    return min(polygon_gap(point, poly, radius) for poly in polygons)


def pd_substeps(position, velocity, raw):
    p, v = np.array(position, float), np.array(velocity, float)
    target = p + 100*np.asarray(raw, float)
    points = []
    for _ in range(10):
        v = v + (100*(target-p)-20*v)*.01
        p = p + v*.01
        points.append(p.copy())
    return np.array(points), v


def audit(directory):
    directory = Path(directory).absolute()
    require(sha(directory/'report.json') == REPORT_SHA, 'fixed actual three-arm receipt')
    bindings = {}
    def read(path):
        h = sha(path)
        require(str(path) not in bindings or bindings[str(path)] == h, 'input drift')
        bindings[str(path)] = h
        return json.loads(path.read_bytes())
    report = read(directory/'report.json')
    require(report['status'] == 'native_c1_three_branch_execution_recorded' and
            [r['arm'] for r in report['rows']] == list(ARMS), 'three complete actual branches')
    checks, rows = {}, []
    for arm in ARMS:
        frames = [read(directory/arm/f'frame_{i:02d}'/'full_public_state.json') for i in range(16)]
        clearances = []
        for public in frames:
            polygons, radius = geometry(public)
            clearances.append(gap(public['bodies']['agent']['position'], polygons, radius))
        max_p_error = max_v_error = 0.
        suffix_substeps = []
        for i in range(1, 16):
            before, after = frames[i-1]['bodies']['agent'], frames[i]['bodies']['agent']
            command = read(directory/arm/f'command_{i:02d}.json')
            checks[f'{arm}_{i}_target'] = np.array_equal(command['absolute_target'],
                np.asarray(before['position'])+100*np.asarray(command['raw_env_argument']))
            points, velocity = pd_substeps(before['position'], before['velocity'], command['raw_env_argument'])
            max_p_error = max(max_p_error, float(np.max(np.abs(points[-1]-after['position']))))
            max_v_error = max(max_v_error, float(np.max(np.abs(velocity-after['velocity']))))
            if i > 10:
                polygons, radius = geometry(frames[i-1])
                suffix_substeps.extend(gap(p, polygons, radius) for p in points)
        blocks = [f['bodies']['block'] for f in frames]
        stationary = all(b['position'] == blocks[10]['position'] and b['angle'] == blocks[10]['angle'] for b in blocks[10:])
        checks[arm+'_pd_positions_exact'] = max_p_error == 0
        checks[arm+'_pd_velocities_exact'] = max_v_error == 0
        checks[arm+'_suffix_stationary'] = stationary
        checks[arm+'_suffix_contact_counts_zero'] = all(f['env']['n_contact_points'] == 0 for f in frames[11:])
        checks[arm+'_positive_substep_clearance'] = min(suffix_substeps) > 0
        rows.append(dict(arm=arm,frame_clearances=clearances,decision_clearance=clearances[10],
            min_reconstructed_suffix_substep_clearance=min(suffix_substeps),
            pd_position_max_abs_error=max_p_error, pd_velocity_max_abs_error=max_v_error,
            object_suffix_stationary=stationary, contacts=[f['env']['n_contact_points'] for f in frames]))
    require(all(sha(p) == h for p, h in bindings.items()), 'inputs changed during computation')
    return dict(schema=NAME,status='saved_native_geometry_consistent' if all(checks.values()) else 'inconsistent',
        checks=checks,all_checks=all(checks.values()),rows=rows,input_bindings=bindings,source_sha256=sha(__file__),
        neural_calls=0,physics_calls=0,P1_admission=False,P5_admission=False,
        scope='One fixed already-opened root; native coordinate units, not resized image pixels. '
              'Substeps algebraically reconstructed, not new physics observations. No outcome-tuned actions or thresholds.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    r = audit(a.execution_dir)
    with a.output.open('x') as f:
        f.write(json.dumps(r, indent=2, sort_keys=True, allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(dict(status=r['status'], all_checks=r['all_checks'])))
    require(r['all_checks'], 'saved geometry mismatch')


if __name__ == '__main__':
    main()
