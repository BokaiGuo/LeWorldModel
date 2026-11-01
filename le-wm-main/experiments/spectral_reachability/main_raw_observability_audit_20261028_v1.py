"""Saved-only observability partition; never a correction or action selector."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = 'main_raw_observability_audit_20261028_v1'
PINS = (
    ('main_raw_coordinate_response_20261028_v1_MAIN01.json',
     '703142be3073336d76cd8cfb9cd8ef8fc41d0c3be0dd7141cc222a635e52c061'),
    ('agent_observed_raw_coordinate_physical_20261028_v1_MAIN01/report.json',
     'c4610c7f495137cbda7d70b147c74670db6f85339561da611ce5bee152d7bb10'),
)
LABELS = ('unobservable_RGB', 'encoder_invariant', 'actual_goal_tie',
          'model_goal_tie', 'same_direction', 'opposite_direction')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def classify(c):
    """Exact ties stay ties; there is no fitted epsilon or favourable h choice."""
    gm, ga = c['model_scalar_fd'], c['actual_scalar_fd']
    require(np.isfinite([gm, ga]).all(), 'finite scalar secants')
    if c['RGB_pixels_identical']:
        require(c['actual_latents_identical'] and ga == 0,
                'identical RGB cannot certify a nonzero real response')
        return 'unobservable_RGB'
    if c['actual_latents_identical']:
        require(ga == 0, 'identical latents imply zero scalar secant')
        return 'encoder_invariant'
    if ga == 0:
        return 'actual_goal_tie'
    if gm == 0:
        return 'model_goal_tie'
    return 'same_direction' if (gm > 0) == (ga > 0) else 'opposite_direction'


def state_response(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    require(a.shape == b.shape == (7,) and np.isfinite([a, b]).all(),
            'whole finite native7 state')
    d = b-a
    pose = bool(np.any(d[:5] != 0))
    velocity = bool(np.any(d[5:] != 0))
    return dict(pose_changed=pose, velocity_changed=velocity,
                velocity_only=bool(velocity and not pose),
                agent_xy_displacement=float(np.linalg.norm(d[:2])),
                object_xy_displacement=float(np.linalg.norm(d[2:4])),
                angle_displacement_wrapped=abs(float(np.arctan2(np.sin(d[4]), np.cos(d[4])))),
                velocity_displacement=float(np.linalg.norm(d[5:])))


def summarize(rows):
    counts = {k: sum(r['label'] == k for r in rows) for k in LABELS}
    require(sum(counts.values()) == len(rows), 'exhaustive disjoint denominator')
    informative = counts['same_direction'] + counts['opposite_direction']
    return dict(pairs=len(rows), labels=counts, informative_direction_pairs=informative,
                opposite_fraction_informative=(counts['opposite_direction']/informative
                                                if informative else None),
                invisible_pose_changes=sum(r['label'] == 'unobservable_RGB' and
                                           r['state']['pose_changed'] for r in rows),
                invisible_velocity_only=sum(r['label'] == 'unobservable_RGB' and
                                             r['state']['velocity_only'] for r in rows))


def build():
    inputs = []
    for name, expected in PINS:
        p = ROOT/'results'/name
        require(p.is_file() and not p.is_symlink() and sha(p) == expected,
                'pinned completed physical/paired report')
        inputs.append(json.loads(p.read_bytes()))
    paired, physical = inputs
    require(paired['available_pairs'] == paired['fixed_pairs'] == 80 and
            paired['P1_admission'] is False and physical['errors'] == [] and
            len(physical['rows']) == 164, 'fixed whole completed cohort')
    rows, panels = [], []
    for p in paired['panels']:
        slot = p['slot']
        require(slot == len(panels)//2, 'original fixed slot order')
        for hi, step in enumerate(p['steps']):
            require(step['h'] == (.01, .001)[hi], 'both fixed h remain separate')
            local = []
            for c in step['comparisons']:
                coord = c['coord']
                require(coord == len(local) and c['available'], 'every fixed coordinate')
                idx = 41*slot+1+20*hi+2*coord
                a, b = physical['rows'][idx:idx+2]
                require(a['index'] == idx and b['index'] == idx+1 and
                        a['plan']['commands'][:5] == b['plan']['commands'][:5],
                        'same original prefix and paired indices')
                ca = [v for v in a['checkpoints'] if v['step'] == 10]
                cb = [v for v in b['checkpoints'] if v['step'] == 10]
                require(len(ca) == len(cb) == 1 and
                        (ca[0]['pixel_sha256'] == cb[0]['pixel_sha256']) == c['RGB_pixels_identical'] and
                        (a['final_state'] == b['final_state']) == c['native_states_identical'],
                        'independent same-time state and pixel identity')
                label = classify(c)
                require((label == 'opposite_direction') == c['strict_direction_reversal'],
                        'original reversal flag independently recomputed')
                row = dict(slot=slot, root_slot=p['root_slot'], condition=p['condition'],
                           h=step['h'], coord=coord, minus_index=idx, plus_index=idx+1,
                           label=label, state=state_response(a['final_state'], b['final_state']))
                rows.append(row)
                local.append(row)
            require(len(local) == 10, 'full ten coordinates')
            panels.append(dict(slot=slot, h=step['h'], **summarize(local)))
    require(len(rows) == 80, 'all80, no omitted ties or invisible probes')
    return dict(schema=NAME, summary=summarize(rows), panels=panels, rows=rows,
                bindings={name: expected for name, expected in PINS},
                known_roots=2, fresh_roots=0, new_neural_calls=0, new_physics_calls=0,
                P1_admission=False, gate_changed=False, correction_deployed=False,
                scope='Exact saved-only stratification, not independent samples or unique causal attribution. '
                      'Invisible pose changes support rendering non-injectivity, not a specific rounding implementation. '
                      'No real gradient or outcome enters a planner.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output).absolute()
    require(out.resolve() == out and out.parent == ROOT/'results' and
            out.name.startswith(NAME+'_'), 'exclusive new own output')
    result = build()
    with out.open('xb') as f:
        f.write(json.dumps(result, sort_keys=True, allow_nan=False).encode()+b'\n')
    print(json.dumps(dict(output=str(out), sha256=sha(out), summary=result['summary'])))


if __name__ == '__main__':
    main()
