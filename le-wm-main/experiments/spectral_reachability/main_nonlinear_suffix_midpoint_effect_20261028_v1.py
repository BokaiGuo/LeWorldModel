"""Predeclared saved-array arithmetic for a fixed-action midpoint intervention.

This module neither runs a model nor authenticates acquisition. The production
adapter must join full strict receipts before calling these pure functions.
No estimated bias, fitted correction, thresholds, selection or controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


NAME = 'main_nonlinear_suffix_midpoint_effect_20261028_v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def vector(value):
    a = np.asarray(value)
    require(a.shape == (192,) and a.dtype.kind in 'fi' and np.isfinite(a).all(),
            'finite native192 vector')
    with np.errstate(over='ignore', invalid='ignore'):
        f = a.astype(np.float32)
    require(np.isfinite(f).all() and np.array_equal(a, f),
            'literal float32 values; never round a foreign coordinate system')
    return f.astype(np.float64)


def mse(a, b):
    return float(np.mean((a-b)**2))


def protocol():
    return dict(
        fixed_rows=4, known_roots=2, fresh_roots=0,
        actions='same full raw10, no reoptimization in this intervention',
        treatment='replace imagined midpoint input with observed RGB history',
        target='same actual terminal RGB and same goal192 in both predictions',
        primary='endpoint MSE to actual terminal encoding, all four rows',
        secondary='absolute error of predicted goal Q versus actual goal Q',
        signed_terms='retain cross term and squared response; do not delete cancellation',
        absent='no baseline ranking repair claim with unmatched conditioning',
        counts='four conditions on two known roots; no independence or significance claim',
        Q='unchanged identity192 mean squared distance',
        training=False, bias_fit=False, selection=False,
        neural_calls=0, physics_calls=0, P1_admission=False,
    )


def compare(imagined_prediction, observed_mid_prediction, actual, goal):
    p, q, a, g = map(vector, (imagined_prediction, observed_mid_prediction, actual, goal))
    e, response = p-a, q-p
    before, after = mse(p, a), mse(q, a)
    pred_before, pred_after, actual_Q = mse(p, g), mse(q, g), mse(a, g)
    cross = float(2*np.mean(e*response))
    square = float(np.mean(response**2))
    delta = after-before
    require(np.isclose(delta, cross+square, rtol=1e-12, atol=1e-12),
            'signed endpoint error identity')
    goal_cross = float(2*np.mean((p-g)*response))
    require(np.isclose(pred_after-pred_before, goal_cross+square, rtol=1e-12, atol=1e-12),
            'signed goal cost identity')
    before_abs, after_abs = abs(pred_before-actual_Q), abs(pred_after-actual_Q)
    return dict(
        imagined_endpoint_MSE=before, observed_mid_endpoint_MSE=after,
        endpoint_MSE_delta=delta,
        endpoint_MSE_better=after < before, endpoint_MSE_worse=after > before,
        imagined_predicted_Q=pred_before, observed_mid_predicted_Q=pred_after,
        actual_Q=actual_Q,
        imagined_absolute_Q_error=before_abs,
        observed_mid_absolute_Q_error=after_abs,
        absolute_Q_error_delta=after_abs-before_abs,
        absolute_Q_error_better=after_abs < before_abs,
        absolute_Q_error_worse=after_abs > before_abs,
        signed_endpoint_terms=dict(cross=cross, response_square=square),
        signed_goal_terms=dict(cross=goal_cross, response_square=square),
        response_MSE=square,
        prediction_changed=bool(np.any(response != 0)),
        physical_action_changed=False,
        physical_benefit_claim=False,
        ranking_repair_claim=False,
    )


def summarize(rows):
    require(len(rows) == 4 and [r['index'] for r in rows] == list(range(4)),
            'fixed ordered four-condition denominator, no dropping')
    require([(r['root_slot'], r['condition']) for r in rows] ==
            [(0, 'off'), (0, 'on'), (1, 'off'), (1, 'on')],
            'exact two roots by off/on, no duplicate panels')
    for r in rows:
        require(r['physical_action_changed'] is False and r['physical_benefit_claim'] is False
                and r['ranking_repair_claim'] is False, 'diagnostic only')
    return dict(
        protocol=protocol(), rows=rows,
        endpoint_counts={key: sum(r[f'endpoint_MSE_{key}'] for r in rows)
                         for key in ('better', 'worse')},
        Q_error_counts={key: sum(r[f'absolute_Q_error_{key}'] for r in rows)
                        for key in ('better', 'worse')},
        independent_confirmation=False, scientific_admission=False,
    )


def seal_protocol(output):
    root = Path(__file__).resolve().parents[2]
    path = Path(output).absolute()
    require(path.resolve() == path and path.parent == root/'results'
            and path.name.startswith(NAME+'_protocol_') and path.suffix == '.json',
            'exclusive own protocol result path')
    sources = [Path(__file__).resolve(), root/'tests'/f'test_{NAME}.py',
               root/'innerdocs'/f'{NAME}.md']
    bindings = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    value = dict(schema=NAME+'.protocol', status='analysis_protocol_only_no_actual_results',
                 protocol=protocol(), bindings=bindings, actual_report=None)
    payload = json.dumps(value, sort_keys=True, allow_nan=False).encode()+b'\n'
    with path.open('xb') as f:
        require(f.write(payload) == len(payload), 'full protocol write')
        f.flush()
        os.fsync(f.fileno())
    return dict(output=str(path), sha256=hashlib.sha256(payload).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    print(json.dumps(seal_protocol(parser.parse_args().output)))
