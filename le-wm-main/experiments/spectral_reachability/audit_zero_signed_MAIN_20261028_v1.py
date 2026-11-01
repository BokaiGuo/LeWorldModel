"""Independent saved arithmetic; no producer, environment or neural imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PINS = {
    'consume_zero_and_rolling_MAIN_20261028_v1_zero_MAIN01.json':
        '7c22af94f7aebaabbe4857cb1797c1b7fe857020145829273fb2349b45ea8e19',
    'consume_mixedmode_capability_MAIN_20261028_v1_MAIN01.json':
        'd1efceb6aa0eb4e72a586bb0c1ba297ecb433f12e048e7a0c966450b71156537',
}
PROBES = ('zero', '+x', '-x', '+y', '-y')
SPACES = {'joint_xy': (0, 1, 2, 3), 'agent_xy': (0, 1),
          'object_xy': (2, 3), 'object_angle_wrapped': (4,)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path, expected):
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, 'external input SHA')
    return json.loads(data)


def delta(a, b, indices):
    d = np.asarray(a, np.float64)[list(indices)] - np.asarray(b, np.float64)[list(indices)]
    return np.arctan2(np.sin(d), np.cos(d)) if indices == (4,) else d


def recompute(zero, selected):
    require(len(zero) == 20 and len(selected) == 40, 'original full denominators')
    selected = [r for r in selected if r['role'] == 'after']
    require(len(selected) == 20, 'predeclared after role, not outcome selection')
    result = []
    for task in range(4):
        arms = [[r for r in rows if r['task'] == task] for rows in (zero, selected)]
        for rows in arms:
            require(tuple(r['probe_id'] for r in rows) == PROBES, 'all fixed probes in order')
            require(all(r['status'] == 'completed' for r in rows), 'no partial endpoint')
        require(all(a['goal'] == b['goal'] and a['root_state'] == b['root_state']
                    and a['condition'] == b['condition'] for a, b in zip(*arms)), 'shared task identity')
        spaces = {}
        for name, indices in SPACES.items():
            direction = delta(arms[0][0]['goal'], arms[0][0]['root_state'], indices)
            norm = np.linalg.norm(direction)
            direction = None if norm == 0 else direction / norm
            values = []
            for rows in arms:
                base = rows[0]['end_state']
                responses = [None if direction is None else float(delta(r['end_state'], base, indices) @ direction)
                             for r in rows]
                # Original exact float32 probe amplitude, held five raw steps.
                amplitude = float(np.float32(.1))
                secants = [None if direction is None else float(
                    delta(rows[p]['end_state'], rows[n]['end_state'], indices) @ direction / (2 * amplitude))
                    for p, n in ((1, 2), (3, 4))]
                values.append(dict(responses=responses, secants=secants))
            changes = lambda key: [None if a is None or b is None else b - a
                                  for a, b in zip(values[0][key], values[1][key])]
            spaces[name] = dict(arms=values, response_changes=changes('responses'),
                                secant_changes=changes('secants'))
        result.append(dict(task=task, condition=arms[0][0]['condition'], spaces=spaces))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--comparison-sha', required=True)
    p.add_argument('--destination', type=Path, required=True)
    args = p.parse_args()
    z, s = [read(ROOT/'results'/name, sha) for name, sha in PINS.items()]
    expected = recompute(z['rows'], s['receipt']['rows'])
    actual = read(args.comparison, args.comparison_sha)
    require(len(actual['comparisons']) == 4 and len(actual['groups']) == 8, 'comparison full denominator')
    checked = 0
    for row, comparison in zip(expected, actual['comparisons']):
        require(row['task'] == comparison['task'] and row['condition'] == comparison['condition'], 'task join')
        for name, space in row['spaces'].items():
            saved = comparison['spaces'][name]
            for key, values in [('response_changes', [v['selected_minus_zero'] for v in saved['signed_response_changes']]),
                                ('secant_changes', saved['signed_secant_changes'])]:
                wanted = space[key]
                require(len(values) == len(wanted), 'complete fixed vector')
                for a, b in zip(values, wanted):
                    require((a is None and b is None) or (a is not None and b is not None and
                            np.isclose(a, b, rtol=1e-12, atol=1e-12)), 'independent signed arithmetic')
                    checked += 1
    result = dict(status='independent_signed_arithmetic_verified_not_admitted',
                  scalar_changes_checked=checked, input_sha256=PINS,
                  comparison_sha256=args.comparison_sha, rows=expected,
                  new_neural_calls=0, new_physics_calls=0, P1_admission=False,
                  limitations='Known roots; elapsed-time control is not state/contact mediation isolation.')
    data = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    with args.destination.open('xb') as stream:
        require(stream.write(data) == len(data), 'complete exclusive write')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(args.destination), sha256=hashlib.sha256(data).hexdigest(),
                          checked=checked, status=result['status'])))


if __name__ == '__main__':
    main()
