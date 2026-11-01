"""Saved-node policy support scan, NOT a new trajectory or full strict consumer."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability import agent_p3_effective_schedule_20261029_v1 as owner

NAME = 'main_p3_policy_support_preflight_20261029_v1'
REPORT = owner.ROOT / 'results/agent_p3_effective_schedule_20261029_v1_MAIN01/report.json'
REPORT_SHA = '840ef006b7d3be039ad84546d2ff8bfe903c3c776a6c6a187fad20c2d4fc7116'
RECEIPT = owner.ROOT / 'results/agent_p3_effective_schedule_20261029_v1_consume_MAIN01.json'
RECEIPT_SHA = 'a48d63d272a4da1b4a270fa3dee60008d1e5d59251146b5ddd8143ba14632b5f'


def route(task, responses, diagnostic):
    """Use exact original same-node signal and choose; never execute an action."""
    owner.require(owner.signal.spectra(task, responses) == diagnostic,
                  'recompute complete spectra before original policy')
    node, signals, seal = owner.signal.runtime_signal(task, responses, diagnostic)
    result = {}
    for mode in ('fixed', 'adaptive'):
        decision = owner.core.choose(task, node, signals, mode)
        supported = (decision['program_type'] == decision['extension_type'] == 'progress'
                     and decision['horizon'] in (1, 3)
                     and decision['continue_expanding'] is True)
        result[mode] = dict(original_decision=decision, horizon_panel_supported=supported,
            implemented_panel_candidates=4,
            original_candidate_quota_executed=False,
            original_depth_executed=False,
            rejection=None if supported else 'unsupported_original_program_type_or_expansion')
    return dict(scheduling_signal=seal['scheduling_signal'], routes=result)


def scan(report):
    owner.require(report['schema'] == owner.NAME + '.report' and
                  report['status'] == 'known_fullcommit_recorded' and not report['errors'],
                  'pinned complete historical report')
    expected = [(n['episode_id'], n['boundary']) for n in owner.node_roster()]
    owner.require([(n['episode_id'], n['boundary']) for n in report['nodes']] == expected,
                  'all historical24 nodes in fixed order')
    rows = []
    for n in report['nodes']:
        r, req = n.get('receipt'), n.get('request')
        row = dict(episode_id=n['episode_id'], boundary=n['boundary'], status=n['status'], policy=None)
        if r is not None:
            owner.require(req is not None and req['episode_id'] == n['episode_id'] and
                          req['boundary'] == n['boundary'], 'same-node request identity')
            row['policy'] = route(req['task'], r['responses'], r['diagnostic'])
        rows.append(row)
    counts = {m: dict(Counter('supported' if r['policy']['routes'][m]['horizon_panel_supported']
                             else 'unsupported' for r in rows if r['policy'] is not None))
              for m in ('fixed', 'adaptive')}
    return dict(schema=NAME + '.scan', rows=rows, route_counts=counts,
        historical_nodes=24, known_roots=2, independent_roots=0,
        source_trajectory_policy='historical fixed commit15/feedback5, NOT the scanned adaptive policy',
        new_trajectory_prediction=False, full_strict_rebuilt_here=False,
        scope='only saved same-node spectra/policy reconstruction; inherited receipt byte-bound separately',
        new_neural_calls=0, new_physics_calls=0, P1_admission=False, P3_admission=False,
        scientific_admission=False, equal_FLOPs=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    for path, expected in ((REPORT, REPORT_SHA), (RECEIPT, RECEIPT_SHA)):
        owner.require(owner.sha(path) == expected, 'pinned historical artifact bytes')
    owner.require(owner.signal.read(RECEIPT)['complete'] is True, 'historical engineering receipt only')
    # The full report contains 24 dense J/trajectory receipts; use its original
    # sealed report cap, not the small helper's default JSON metadata limit.
    result = scan(owner.signal.read(REPORT, owner.contract()['final_reserve_bytes']))
    own = [Path(__file__).resolve(), owner.ROOT/'tests'/('test_'+NAME+'.py')]
    result['bindings'] = {str(x): owner.sha(x) for x in own + [REPORT, RECEIPT,
        Path(owner.__file__).resolve(), Path(owner.signal.__file__).resolve(), Path(owner.core.__file__).resolve()]}
    output = Path(args.output).absolute()
    owner.require(output.parent == owner.ROOT/'results' and output.resolve() == output and
                  output.name.startswith(NAME+'_') and output.suffix == '.json', 'exclusive own new output')
    data = owner.raw(result)
    owner.require(len(data) <= 4*1024**2, 'bounded saved scan')
    with output.open('xb') as f:
        owner.require(f.write(data) == len(data), 'whole scan write')
        f.flush()
        os.fsync(f.fileno())
    owner.signal.fsync_dir(output.parent)
    print(json.dumps(dict(output=str(output), sha256=hashlib.sha256(data).hexdigest(),
                          route_counts=result['route_counts']), sort_keys=True))


if __name__ == '__main__':
    main()
