"""Saved feature-space substitutions, NOT simulator interventions or a planner."""
import argparse
import json
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import main_observed_raw_fd_recompute_20261028_v1 as model
from experiments.spectral_reachability import main_raw_observability_audit_20261028_v1 as observable

ROOT = model.ROOT
NAME = 'main_raw_anchor_factorial_20261028_v1'
require = model.require
PINS = (
    (ROOT/'results/main_raw_coordinate_response_20261028_v1_MAIN01.json',
     '703142be3073336d76cd8cfb9cd8ef8fc41d0c3be0dd7141cc222a635e52c061'),
    (ROOT/'results/agent_observed_raw_coordinate_rgb_20261028_v1_consume_MAIN01.json',
     'd6b6ed08732d7595e740f98e0dc2450adf439e726aa3078854badb78f7d8b4b2'),
)


def label(g, actual):
    require(np.isfinite([g, actual]).all(), 'finite scalar direction')
    if actual == 0:
        return 'actual_tie'
    if g == 0:
        return 'substituted_tie'
    return 'same' if (g > 0) == (actual > 0) else 'opposite'


def factorial(pm, pp, am, ap, pbase, abase, goal, span):
    """Four exact anchored quadratic secants; neither factor is fitted.

    First letter = anchor M/A; second = action-response M/A. M=model,
    A=actual. AM is an oracle common-offset subtraction from model endpoints;
    MA substitutes actual responses at the model anchor. No term is omitted.
    """
    pm, pp, am, ap, pb, ab, g = [np.asarray(v, float) for v in
                                (pm, pp, am, ap, pbase, abase, goal)]
    require(all(v.shape == (192,) and np.isfinite(v).all() for v in (pm, pp, am, ap, pb, ab, g))
            and type(span) in (int, float) and np.isfinite(span) and span > 0,
            'whole finite native192 and positive actual span')
    def secant(anchor, minus, plus):
        return float((np.mean((anchor+plus-g)**2)-np.mean((anchor+minus-g)**2))/span)
    dm, dp, da, db = pm-pb, pp-pb, am-ab, ap-ab
    # Compute original routes directly, avoiding an unnecessary add/subtract roundtrip.
    mm = float((np.mean((pp-g)**2)-np.mean((pm-g)**2))/span)
    aa = float((np.mean((ap-g)**2)-np.mean((am-g)**2))/span)
    ma, aim = secant(pb, da, db), secant(ab, dm, dp)
    anchor_effect = ma-aa
    response_effect = aim-aa
    interaction = mm-ma-aim+aa
    direct_interaction = 2*float(np.mean((pb-ab)*((pp-pm)-(ap-am))))/span
    require(abs(interaction-direct_interaction) <= 1e-9*(1+abs(mm)+abs(ma)+abs(aim)+abs(aa)),
            'whole anchor/response factorial interaction identity')
    return dict(secants=dict(MM=mm, AM=aim, MA=ma, AA=aa),
                signs={k: label(v, aa) for k, v in dict(MM=mm, AM=aim, MA=ma).items()},
                baseline_anchor_effect=anchor_effect, baseline_response_effect=response_effect,
                interaction=interaction, interaction_identity_residual=interaction-direct_interaction,
                response_error_L2=float(np.linalg.norm(((pp-pm)-(ap-am))/span)))


def counts(rows):
    return {route: {name: sum(r['factorial']['signs'][route] == name for r in rows)
                    for name in ('same', 'opposite', 'actual_tie', 'substituted_tie')}
            for route in ('MM', 'AM', 'MA')}


def build():
    allpins = (*model.PINS, *PINS)
    for p, h in allpins:
        require(observable.sha(p) == h, 'exact completed model/actual/paired bytes')
    prepared, report, strict, paired, rgb = [json.loads(p.read_bytes()) for p, _ in allpins]
    require(strict['complete'] and report['errors'] == [] and paired['available_pairs'] == 80
            and len(rgb['rows']) == 164 and rgb['fixed_denominator'] == 164,
            'full fixed actual/model cohort, no padding or selection')
    rows, panels = [], []
    for slot in range(4):
        values = report['rows'][slot]['values']
        base = 41*slot
        pb = values[0]['context']['latent_history'][-1]
        ab = rgb['rows'][base]['actual192']
        for hi, h in enumerate((.01, .001)):
            local = []
            for coord in range(10):
                k = 1+20*hi+2*coord
                idx = base+k
                jobs = prepared['jobs'][idx:idx+2]
                accepted = rgb['rows'][idx:idx+2]
                c = paired['panels'][slot]['steps'][hi]['comparisons'][coord]
                require(c['coord'] == coord and c['available'] and
                        all(j['index'] == a['index'] == i and j['id'] == a['probe_id'] and
                            j['job_sha256'] == a['producer_job_sha256'] and
                            a['goal192'] == j['task']['goal'] and
                            (j['coord'], j['h'], j['sign']) == (coord, h, sign)
                            for j, a, i, sign in zip(jobs, accepted, (idx, idx+1), (-1, 1), strict=True)),
                        'exact paired indices/actions/goals and all scales')
                pm, pp = [v['context']['latent_history'][-1] for v in values[k:k+2]]
                f = factorial(pm, pp, accepted[0]['actual192'], accepted[1]['actual192'],
                              pb, ab, jobs[0]['task']['goal'], jobs[0]['actual_pair_spacing'])
                require(f['secants']['MM'] == c['model_scalar_fd'] and
                        f['secants']['AA'] == c['actual_scalar_fd'], 'original scalar endpoints reproduced')
                r = dict(slot=slot, root_slot=jobs[0]['root_slot'], condition=jobs[0]['condition'],
                         h=h, coord=coord, observable_label=observable.classify(c), factorial=f)
                rows.append(r)
                local.append(r)
            panels.append(dict(slot=slot, h=h, counts=counts(local)))
    require(len(rows) == 80, 'all eighty related coordinate probes')
    return dict(schema=NAME, rows=rows, panels=panels, counts=counts(rows),
                original_reversals=sum(r['factorial']['signs']['MM'] == 'opposite' for r in rows),
                original_reversals_still_opposite_after_oracle_centering=sum(
                    r['factorial']['signs']['MM'] == r['factorial']['signs']['AM'] == 'opposite' for r in rows),
                new_opposites_after_oracle_centering=sum(
                    r['factorial']['signs']['MM'] != 'opposite' and
                    r['factorial']['signs']['AM'] == 'opposite' for r in rows),
                bindings={str(p): h for p, h in allpins}, known_roots=2, fresh_roots=0,
                new_neural_calls=0, new_physics_calls=0, P1_admission=False,
                planner_changed=False, deployable=False,
                scope='Post-outcome feature-space factorial arithmetic on fixed all80. Oracle anchor and '
                      'actual responses unavailable to a blind planner. Not new physics, causal system '
                      'interventions, learned corrections, independent evidence or unique root cause.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output).absolute()
    require(out.resolve() == out and out.parent == ROOT/'results' and out.name.startswith(NAME+'_'),
            'exclusive new own output')
    result = build()
    with out.open('xb') as f:
        f.write(json.dumps(result, sort_keys=True, allow_nan=False).encode()+b'\n')
    print(json.dumps(dict(output=str(out), sha256=observable.sha(out), counts=result['counts'],
        original_reversals_still_opposite_after_oracle_centering=result['original_reversals_still_opposite_after_oracle_centering'],
        new_opposites_after_oracle_centering=result['new_opposites_after_oracle_centering'])))


if __name__ == '__main__':
    main()
