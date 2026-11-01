"""Independent saved-only paired native analysis; no model or simulator calls."""
import argparse
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_observed_CEM_physical_20261028_v2 as source

NAME = 'analyze_observed_CEM_MAIN_20261028_v1'


def metrics(state, goal):
    x, g = np.asarray(state, float), np.asarray(goal, float)
    source.require(x.shape == g.shape == (7,) and np.isfinite(x).all() and np.isfinite(g).all(), 'native7 finite')
    angle = min(abs(g[4]-x[4]), 2*np.pi-abs(g[4]-x[4]))
    xy = float(np.linalg.norm(g[:4]-x[:4]))
    return dict(joint_xy_error=xy, object_xy_error=float(np.linalg.norm(g[2:4]-x[2:4])),
                native_state7_distance=float(np.linalg.norm(g-x)), native_angle_term=float(angle),
                native_success=bool(xy < 20 and angle < np.pi/9))


def paired(rows, prefixes, old_rows):
    order = source.ORDER
    source.require([(r['root_slot'], r['prefix_arm']) for r in rows] == order and
                   [(r['root_slot'], r['arm']) for r in prefixes] == order and
                   [(r['root_slot'], r['arm']) for r in old_rows] == order, 'complete fixed4 order')
    out = []
    for row, ref, old in zip(rows, prefixes, old_rows, strict=True):
        source.require(row['prefix_exact'] is row['prefix_goal_exact'] is True and
                       row['goal'] == ref['goal'] == old['goal'] and
                       old['selection_digest'] == ref['original_selection_digest'], 'exact paired prefix/goal/selection')
        before, after = metrics(old['final_state'], old['goal']), metrics(row['final_state'], row['goal'])
        source.require(before['native_success'] == old['final_metrics']['native_success'] and
                       after['native_success'] == row['native_success'] == row['final_metrics']['native_success'], 'independent native success reconciliation')
        source.require(row['status'] in ('horizon_completed', 'native_success', 'native_stopped'), 'all4 actual returned suffixes')
        root = metrics(ref['prefix'][0]['state'], row['goal'])
        prefix = metrics(ref['prefix'][5]['state'], row['goal'])
        out.append(dict(root_slot=row['root_slot'], prefix_arm=row['prefix_arm'], old=before, CEM=after,
                        delta_joint_xy=after['joint_xy_error']-before['joint_xy_error'],
                        delta_object_xy=after['object_xy_error']-before['object_xy_error'],
                        prefix_joint_xy_change=prefix['joint_xy_error']-root['joint_xy_error'],
                        returned_suffix_steps=row['own_returned']-5,
                        terminated=row['terminated'], truncated=row['truncated']))
    return dict(rows=out, denominator=4, old_native_successes=sum(r['old']['native_success'] for r in out),
                CEM_native_successes=sum(r['CEM']['native_success'] for r in out),
                mean_delta_joint_xy=float(np.mean([r['delta_joint_xy'] for r in out])),
                mean_delta_object_xy=float(np.mean([r['delta_object_xy'] for r in out])))


def run(directory, report_sha, output):
    dest = Path(output).absolute()
    source.require(not dest.exists() and dest.parent == source.ROOT/'results' and dest.name.startswith(NAME), 'exclusive analysis output')
    consumed = source.consume(directory, report_sha)
    source.require(consumed['status'] == 'saved_physical_verified_not_admitted', 'complete strict physical acceptance')
    binding = source.read(Path(directory)/'execution_binding.json')
    p = source.verify(binding['prepared'], binding['prepare_sha256'])
    args = p['arguments']
    old = source.read(args['original_physical_report'])
    result = paired(consumed['rows'], p['prefixes'], old['rows'])
    result.update(schema=NAME+'.report', physical_report_sha256=report_sha,
                  original_report_sha256=args['original_physical_report_sha256'],
                  strict_consume=consumed, source_sha256=source.sha(__file__),
                  tests_sha256=source.sha(source.ROOT/'tests'/('test_'+NAME+'.py')),
                  known_roots=2, independent_confirmation=False, P1_admission=False,
                  equal_total_compute=False, equal_FLOPs=False, new_neural_calls=0, new_physics_calls=0,
                  caveats=['Known-root descriptive calibration; no statistical confirmation.',
                           'CEM suffix vs historical suffix, not full-policy or equal-compute comparison.',
                           'Native stops are retained without padding; state7 distance includes velocity but native success does not.',
                           'A successful suffix does not prove a non-greedy capability-changing prefix.'])
    payload = source.raw(result)
    source.require(len(payload) < 2*1024**2, 'bounded full saved consume and analysis')
    with dest.open('xb') as f:
        source.require(f.write(payload) == len(payload), 'full analysis write'); f.flush(); os.fsync(f.fileno())
    fd = os.open(dest.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return dict(path=str(dest), sha256=source.sha(dest), CEM_native_successes=result['CEM_native_successes'],
                old_native_successes=result['old_native_successes'], mean_delta_joint_xy=result['mean_delta_joint_xy'],
                P1_admission=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('input', 'report-sha', 'output'): parser.add_argument('--'+key, required=True)
    a = parser.parse_args()
    print(source.raw(run(a.input, a.report_sha, a.output)).decode())
