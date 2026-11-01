"""Independent saved native-metric arithmetic; no planner/consumer imports."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUTS = {
    'cem': ('agent_moving_CEM8_physical_20261028_v1_MAIN01/report.json',
            'f359812367d8e901b171921552ee464cb3be399168e825bdca4c3fdde4c0a2aa'),
    'spectral': ('capability_suffix_factorial_physical_20261027_v1_MAIN01/report.json',
                 'cc92465141b400659e024e052b05683153b4dc6c013b81bdf1e3e29355fd2663'),
    'analysis': ('agent_CEM8_physical_analysis_20261028_v1_MAIN01.json',
                 'a3f874328acec265428c95c209aed3dc3fb5fcaebb2f822a3e6daeec5ac4d370'),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(row):
    require(row['status'] in ('suffix_completed', 'native_stopped'), 'complete actual endpoint')
    require(row['suffix_returned'] == len(row['responses']) and row['responses'], 'actual steps')
    values = []
    for step, response in enumerate(row['responses'], 1):
        require(response['step'] == step, 'contiguous actual responses')
        s, g = np.asarray(response['state'], np.float64), np.asarray(row['goal'], np.float64)
        require(s.shape == g.shape == (7,) and np.isfinite(s).all() and np.isfinite(g).all(), 'finite native7')
        d = s-g
        a = float(abs(d[4]))
        success = bool(np.linalg.norm(d[:4]) < 20 and min(a, 2*np.pi-a) < np.pi/9)
        v = dict(joint_xy_error=float(np.linalg.norm(d[:4])),
                 native7_distance=float(np.linalg.norm(d)), native_success=success)
        require(response['reward'] == -v['native7_distance'] and response['native_success'] is success,
                'native reward/success independently recomputed')
        require(response['terminated'] is success and not response['truncated'], 'actual terminal semantics')
        if step < len(row['responses']):
            require(not success, 'no steps after native success')
        values.append(v)
    require(row['final_state'] == row['responses'][-1]['state'], 'actual endpoint, never padded H10')
    return values[-1]


def main():
    docs, bindings = {}, {}
    for key, (relative, expected) in INPUTS.items():
        path = ROOT/'results'/relative
        require(sha(path) == expected, 'external actual input SHA: '+key)
        docs[key] = json.loads(path.read_bytes()); bindings[str(path)] = expected
    c, s, analysis = docs['cem'], docs['spectral'], docs['analysis']
    require(len(c['rows']) == 8 and len(s['rows']) == 24 and len(analysis['rows']) == 8,
            'complete original denominators')
    pairs = []
    for i, (slot, condition, arm) in enumerate(
            (slot, cond, arm) for slot in (0, 1) for cond in ('off', 'on')
            for arm in ('line_search_GN', 'full10_conditional_D2P')):
        cr, sr, ar = c['rows'][i], s['rows'][3*i+1], analysis['rows'][i]
        require((cr['root_slot'], cr['condition'], cr['prefix_arm']) ==
                (sr['root_slot'], sr['condition'], sr['arm']) == (slot, condition, arm), 'fixed identity join')
        require(sr['variant'] == 'teacher_full' and cr['goal'] == sr['goal'] and
                cr['start_state'] == sr['start_state'] and cr['selection_digest'] == sr['selection_digest'],
                'same task, same literal prefix ancestry')
        cm, sm = metric(cr), metric(sr)
        for endpoint, computed in ((ar['cem'], cm), (ar['spectral_teacher_full'], sm)):
            require(endpoint['endpoint']['joint_xy_error'] == computed['joint_xy_error'] and
                    endpoint['endpoint']['native7_distance'] == computed['native7_distance'] and
                    endpoint['endpoint']['native']['native_success'] is computed['native_success'],
                    'independent endpoint matches analyzed value')
        pairs.append(dict(index=i, root_slot=slot, condition=condition, prefix_arm=arm,
                          cem=cm, spectral=sm,
                          CEM_minus_spectral_joint_xy=cm['joint_xy_error']-sm['joint_xy_error']))
    delta = float(np.mean([p['CEM_minus_spectral_joint_xy'] for p in pairs]))
    require(delta == analysis['summaries']['all']['paired_difference']['endpoint']
            ['mean_CEM_minus_spectral']['joint_xy_error'], 'full8 mean, not selected subset')
    result = dict(schema='MAIN.CEM8_pair_independent.v1', input_sha256=bindings, rows=pairs,
                  fixed_anchor_count=8, known_roots=2, new_independent_roots=0,
                  mean_CEM_minus_spectral_joint_xy=delta,
                  CEM_lower_joint_xy_count=sum(p['CEM_minus_spectral_joint_xy'] < 0 for p in pairs),
                  CEM_success_indices=[p['index'] for p in pairs if p['cem']['native_success']],
                  spectral_success_indices=[p['index'] for p in pairs if p['spectral']['native_success']],
                  no_new_neural_or_physics=True, equal_total_compute=False, P1_admission=False)
    for path, expected in bindings.items():
        require(sha(Path(path)) == expected, 'input unchanged')
    output = ROOT/'results/recompute_CEM8_pair_MAIN_20261028_v1_MAIN01.json'
    body = (json.dumps(result, sort_keys=True, allow_nan=False)+'\n').encode()
    with output.open('xb') as stream:
        stream.write(body); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(output), sha256=sha(output),
                          mean_CEM_minus_spectral_joint_xy=delta)))


if __name__ == '__main__':
    main()
