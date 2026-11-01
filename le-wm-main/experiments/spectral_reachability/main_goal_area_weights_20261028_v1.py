"""Goal-only exact-fill area integration; a new calibration metric, not a gate fix.

No candidate features/outcomes enter this producer. Whole-patch all196 remains
empty in the older contract; this separate area rule is not its relaxation.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import agent_p5_goal_object_interior_contract_20261028_v1 as goal

NAME = 'main_goal_area_weights_20261028_v1'
ROOT = goal.ROOT
INPUT = ROOT/'results/agent_p5_goal_object_interior_contract_20261028_v1_prepare_MAIN01.json'
INPUT_SHA = 'dbe0ba77d8cf18777589f529a5aedc30c4951ff479fb96bc14a4c5c72ae182aa'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def area(counts):
    require(type(counts) is list and len(counts) == 256 and
            all(type(n) is int and 0 <= n <= 196 for n in counts), 'all256 literal pixel counts')
    total = sum(counts)
    require(total > 0, 'empty visible support: no substitute or zero metric')
    weights = [Fraction(n, total) for n in counts]
    require(sum(weights, Fraction()) == 1, 'exact normalized area')
    return dict(pixel_count=total, all256_counts=counts,
        weights_rational=[dict(numerator=str(x.numerator), denominator=str(x.denominator)) for x in weights],
        weights_float64=[float(x) for x in weights],
        sqrt_diagonal_for_original_flat_MSE=[float(np.sqrt(256*float(x))) for x in weights],
        formula='w_p=count_p/sum_counts; cost=sum_p w_p mean_channel((z_p-goal_p)^2)',
        operator='For original mean over256*384 coordinates, Q repeats sqrt(256*w_p) over384 channels.',
        mask_semantics='visible exact body-fill RGB support, not full object geometry or causal token locality')


def weighted_cost(features, target, weights):
    """Arithmetic API only; caller must seal weights before loading candidates."""
    a, b = np.asarray(features), np.asarray(target)
    w = np.asarray(weights, np.float64)
    require(a.dtype == b.dtype == np.float32 and a.shape == b.shape == (256, 384)
            and np.isfinite(a).all() and np.isfinite(b).all(), 'native finite patch arrays')
    require(w.shape == (256,) and np.isfinite(w).all() and (w >= 0).all() and (w > 0).any(), 'finite nonempty weights')
    return float(np.dot(w, np.mean((a.astype(np.float64)-b.astype(np.float64))**2, axis=1)))


def build():
    require(sha(INPUT) == INPUT_SHA, 'fixed goal-only source')
    previous = goal.read_json(INPUT, INPUT_SHA)
    require(canonical(previous) == canonical(goal.prepare()), 'complete goal-only source reconstruction')
    result = area(previous['support']['all256_pixel_counts'])
    require(result['pixel_count'] == previous['support']['pixel_support']['count'], 'same exact goal pixels')
    result.update(schema=NAME, status='goal_only_area_rule_sealed_NOT_validated',
        source=dict(path=str(INPUT), sha256=INPUT_SHA),
        own_sources={str(p): sha(p) for p in (Path(__file__), ROOT/'tests'/f'test_{NAME}.py')},
        rationale='Exact visible-pixel area integration across the fixed patch grid; no all196 threshold or top-k.',
        preregistration_boundary='New hypothesis informed by goal-only support feasibility; not preregistered before historical candidate results.',
        limitations=['resize color coincidences unresolved', 'global attention is not local causality',
                     'partial visible body support is not complete geometry', 'no independent efficacy evidence'],
        forbidden=['candidate-dependent mask', 'coverage-selected weights', 'threshold sweep', 'fitted Q', 'old gate replacement'],
        encoder_calls=0, predictor_calls=0, physics_calls=0, fitting=False,
        calibration_candidate_only=True, scientific_admission=False, P1_admission=False, P5_admission=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify'))
    parser.add_argument('--output', required=True); parser.add_argument('--sha256')
    args = parser.parse_args(); path = Path(args.output).absolute()
    require(path.resolve() == path and path.parent == ROOT/'results' and path.name.startswith(NAME+'_'), 'own exclusive path')
    value = build()
    if args.command == 'prepare':
        with path.open('xb') as stream:
            stream.write(canonical(value)+b'\n'); stream.flush(); os.fsync(stream.fileno())
    else:
        require(sha(path) == args.sha256 and canonical(json.loads(path.read_bytes())) == canonical(value), 'entire own reconstruction')
    print(json.dumps(dict(path=str(path), sha256=sha(path), pixel_count=value['pixel_count'], scientific_admission=False)))


if __name__ == '__main__':
    main()
