"""Separate nominal distance from modeled ridge improvement on the fixed panel."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import p1_fixed40_conditional_reach_consume_20261029 as consumer

ROOT = consumer.host.ROOT
SOURCE = ROOT/'results/p1_fixed40_conditional_reach_host_20261029_MAIN01.json'
SHA = '9281d3353c8a36bcf7d836614a3142afbc73321064efb68edd52ce2533e9439c'


def gain(nominal, goal, reach):
    n, g = [np.asarray(x, dtype=np.float64) for x in (nominal, goal)]
    if n.shape != (192,) or g.shape != (192,) or not np.isfinite(n).all() or not np.isfinite(g).all() or not np.isfinite(reach):
        raise ValueError('finite nominal, goal and reach required')
    zero = float(np.mean((n-g)**2))
    improvement = zero-float(reach)
    if improvement < -1e-9:
        raise ValueError('ridge optimum worse than zero feasible action')
    return dict(zero_action_Q=zero, ridge_objective=float(reach), regularized_gain=improvement)


def build():
    consumer.verify(SOURCE, SHA)
    report = json.loads(SOURCE.read_bytes())
    h = consumer.host.prefix.history
    prep = h.rgb.read(h.TEACHER_PREP, h.TEACHER_PREP_SHA)
    rows = []
    for row, anchor in zip(report['rows'], prep['anchors'], strict=True):
        if row['anchor_id'] != anchor['anchor_id'] or row['task_id'] != anchor['task_id']:
            raise ValueError('source identity mismatch')
        rows.append(dict(anchor_index=row['anchor_index'], task_id=row['task_id'], cases={
            k:gain(row['cases'][k]['nominal'], anchor['goal'], row['cases'][k]['solve']['own_reach_value'])
            for k in ('predicted','observed')}))
    pairs = []
    for i in range(0,8,2):
        if rows[i]['task_id'] != rows[i+1]['task_id']:
            raise ValueError('cross task pair')
        pairs.append(dict(indices=[i,i+1], cases={k:dict(
            S_minus_zero_gain=rows[i+1]['cases'][k]['regularized_gain']-rows[i]['cases'][k]['regularized_gain'],
            S_minus_zero_nominal=rows[i+1]['cases'][k]['zero_action_Q']-rows[i]['cases'][k]['zero_action_Q'])
            for k in ('predicted','observed')}))
    return dict(schema='p1_fixed40_reach_gain_20261029',source_sha256=SHA,rows=rows,pairs=pairs,
                P1_admission=False,neural_calls=0,physical_calls=0,
                scope='fixed known roots, linear ridge proxy only, no real intervention')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not a.output.is_absolute() or a.output.parent != ROOT/'results' or a.output.exists():
        raise ValueError('exclusive absolute output required')
    print('Separate model ridge gain from nominal goal distance; no model or physical calls.',flush=True)
    data = json.dumps(build(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with a.output.open('xb') as f:
        f.write(data)
    print(json.dumps(dict(path=str(a.output),sha256=hashlib.sha256(data).hexdigest())))


if __name__ == '__main__':
    main()
