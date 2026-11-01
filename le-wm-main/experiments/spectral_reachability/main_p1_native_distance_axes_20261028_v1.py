"""Saved-only native-distance axis attribution; no replacement metric or policy."""
import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT/'results/agent_p1_model_library_selector_evaluation_20261028_v1_MAIN01.json'
EVALUATION_SHA = 'd8b3bedd38e9f9a56d0156f2452d5a20f363acbdad42ea5e3ee4d91f9626ff67'
WRAPPER = ROOT/'external/jepa-wms/evals/simu_env_planning/envs/pusht_gym_wrap.py'
WRAPPER_SHA = 'ed4689253c571c66250503bfaea4c67f3accf88228141096d7de7971f5215942'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sign(x):
    return int(x > 0)-int(x < 0)


def exact_axes(state, goal):
    require(len(state) == len(goal) == 7 and all(type(x) in (int, float) and math.isfinite(x)
            for x in [*state, *goal]), 'finite complete native state7')
    d = [Fraction.from_float(float(x))-Fraction.from_float(float(y)) for x, y in zip(state, goal)]
    return dict(xy=sum((x*x for x in d[:4]), Fraction()), angle=d[4]**2,
                velocity=d[5]**2+d[6]**2)


def pair(a, b):
    terms = {k: a['_axes'][k]-b['_axes'][k] for k in ('xy', 'angle', 'velocity')}
    pose, total = terms['xy']+terms['angle'], sum(terms.values(), Fraction())
    latent = None if a['actual_Q'] is None or b['actual_Q'] is None else sign(a['actual_Q']-b['actual_Q'])
    return dict(i=a['index'], j=b['index'], latent_sign=latent,
                squared_native_sign=sign(total), squared_xy_sign=sign(terms['xy']),
                squared_pose_sign=sign(pose), velocity_sign=sign(terms['velocity']),
                squared_margin_exact={k: [str(v.numerator), str(v.denominator)] for k, v in terms.items()},
                latent_native_reversal=None if latent is None else latent*sign(total) < 0,
                latent_xy_reversal=None if latent is None else latent*sign(terms['xy']) < 0,
                latent_pose_reversal=None if latent is None else latent*sign(pose) < 0,
                velocity_flips_pose_order=sign(pose)*sign(total) < 0,
                native_float64_sign_equal_exact=sign(a['physical_metrics']['native_distance']-
                                                     b['physical_metrics']['native_distance']) == sign(total))


def build():
    require(sha(EVALUATION) == EVALUATION_SHA and sha(WRAPPER) == WRAPPER_SHA, 'pinned saved authority and wrapper')
    value = json.loads(EVALUATION.read_bytes())
    rows = value['rows']
    require(len(rows) == 40 and [r['index'] for r in rows] == list(range(40)), 'fixed all40')
    for row in rows:
        require((row['actual_Q'] is None) == (row['index'] == 12), 'original unknown only')
        row['_axes'] = exact_axes(row['returned_state'], row['physical_goal'])
        # Preserve the actual evaluator, including its unwrapped distance angle and velocity.
        require(math.isclose(math.sqrt(float(sum(row['_axes'].values(), Fraction()))),
                             row['physical_metrics']['native_distance'], rel_tol=2e-15), 'saved native distance recomputation')
    pairs = []
    for anchor in range(8):
        group = rows[anchor*5:anchor*5+5]
        require(all(r['physical_goal'] == group[0]['physical_goal'] and
                    r['actual_return_absolute_step'] == 20 for r in group), 'same goal and same return20')
        pairs.extend(pair(a, b) for a, b in itertools.combinations(group, 2))
    available = [p for p in pairs if p['latent_sign'] is not None]
    require(len(pairs) == 80 and len(available) == 76 and all(p['native_float64_sign_equal_exact'] for p in pairs),
            '80/76 denominator and exact native ordering')
    return dict(schema='main_p1_native_distance_axes_20261028_v1',
                status='saved_axis_attribution_not_scientific_admission',
                authority=dict(evaluation=str(EVALUATION), evaluation_sha256=EVALUATION_SHA,
                               wrapper=str(WRAPPER), wrapper_sha256=WRAPPER_SHA,
                               upstream_rebuilt_in_this_tool=False),
                fixed_pairs=80, comparable_latent_pairs=76, unknown_pairs=4, known_roots=2, fresh_roots=0,
                summary=dict(latent_native_reversals=sum(p['latent_native_reversal'] for p in available),
                    latent_xy_reversals=sum(p['latent_xy_reversal'] for p in available),
                    latent_pose_reversals=sum(p['latent_pose_reversal'] for p in available),
                    velocity_order_flips_all80=sum(p['velocity_flips_pose_order'] for p in pairs),
                    native_reversals_concordant_with_pose_and_flipped_by_velocity=sum(
                        p['latent_native_reversal'] and p['latent_sign']*p['squared_pose_sign'] > 0 and
                        p['velocity_flips_pose_order'] for p in available)),
                pairs=pairs, new_neural_calls=0, new_physics_calls=0, gate_changed=False,
                policy_or_metric_changed=False, P1_admission=False,
                interpretation='Exact state7 distance attribution, not a replacement for native success, a causal intervention, or a fitted image metric.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).absolute()
    require(output.resolve() == output and output.parent == ROOT/'results' and
            output.name.startswith('main_p1_native_distance_axes_20261028_v1_'), 'exclusive own result path')
    result = build()
    raw = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n'
    with output.open('xb') as stream:
        stream.write(raw)
    print(json.dumps(dict(output=str(output), sha256=sha(output), summary=result['summary'])))


if __name__ == '__main__':
    main()
