"""Independent saved native-state 2x2 arithmetic; no model/physics imports.

Reuses ONLY exact full action identities at the same task/root/prefix. Receipt
acceptance is explicit pinned authority, not a new upstream physical audit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = 'main_anchor_guard_physical_factorial_20261028_v1'
AXES = ('joint_xy', 'agent_xy', 'object_xy', 'velocity', 'angle', 'native_distance')
PINS = {
    'cells': ('agent_p2_safeguard_factorial_cuda_20261028_v1_consume_MAIN01.json',
              '8a59a189feaedf3b215b7aa9245c3b027228e2a51cfef32a47df286c664e5d89'),
    'control': ('agent_p2_contextual_composition_physical_handoff_20261028_v1_MAIN01/report.json',
                'c9e8c37984926a68bf2e4d3a92fe47737c03e719ed51917da58b72c888eb038d'),
    'incumbent': ('agent_p2_contextual_midfeedback_recompile_20261028_v2_physical_MAIN01/report.json',
                  '039e23ce87980e5a66f38cb47bc50c106690d17d2adb7169a4d4bed0adb70956'),
    'incumbent_strict': ('agent_p2_contextual_midfeedback_recompile_20261028_v2_physical_consume_MAIN01.json',
                         '7e71cf504691cb1b132c437267319f3291a62b0f6c9f839f0652b42a974f8de8'),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and sha(path) == expected, 'pinned regular input')
    return json.loads(path.read_bytes())


def bits(commands):
    a = np.asarray(commands, dtype=np.float64)
    require(a.shape == (10, 2) and np.isfinite(a).all() and (np.abs(a) < 1).all()
            and np.array_equal(a, a.astype(np.float32)), 'literal legal full10 float32')
    return a.astype('<f4').tobytes().hex()


def metrics(row):
    s, g = [np.asarray(row[k], dtype=np.float64) for k in ('final_state', 'goal')]
    require(s.shape == g.shape == (7,) and np.isfinite(s).all() and np.isfinite(g).all(), 'finite native state7')
    d = s-g
    a = abs(float(d[4]))
    return dict(joint_xy=float(np.linalg.norm(d[:4])), agent_xy=float(np.linalg.norm(d[:2])),
        object_xy=float(np.linalg.norm(d[2:4])), velocity=float(np.linalg.norm(d[5:])),
        angle=abs(float(np.arctan2(np.sin(d[4]), np.cos(d[4])))),
        native_distance=float(np.linalg.norm(d)),
        native_success=bool(np.linalg.norm(d[:4]) < 20 and min(a, 2*np.pi-a) < np.pi/9))


def assemble(cells, reports, plans):
    order = [(i, a, g) for i in range(8) for a in ('zero', 'incumbent') for g in (False, True)]
    require(len(cells) == 32 and [(c['branch'], c['anchor'], c['guard']) for c in cells] == order,
            'entire fixed32 factorial, ordered no subset')
    require(set(reports) == set(plans) == {'control', 'zero', 'incumbent'}, 'three literal execution cohorts')
    for name in reports:
        require(len(reports[name]) == len(plans[name]) == 8, 'all8 native denominators')
        for i, (r, p) in enumerate(zip(reports[name], plans[name], strict=True)):
            require(r['index'] == i and r['own_returned'] == 10 and r['status'] == 'horizon_completed',
                    'complete full10 only, early stops not padded')
            require(all(r[k] == p[k] for k in ('root_slot', 'condition', 'arm', 'task_id', 'selection_digest')),
                    'native row bound to its literal dispatched plan')
    rows = []
    for c in cells:
        i = c['branch']; reference = reports['control'][i]
        require(bits(c['full10']) == c['full10_raw_bits'] and c['prefix_raw_bits'] == c['full10_raw_bits'][:80],
                'sealed factorial full10/prefix bits')
        # Routing depends only on fixed factor and exact raw action; no outcomes.
        source = 'incumbent' if c['anchor'] == 'incumbent' else 'zero'
        if source == 'zero' and c['guard'] and bits(plans['zero'][i]['commands']) != c['full10_raw_bits']:
            source = 'control'
        row, plan = reports[source][i], plans[source][i]
        require(bits(plan['commands']) == c['full10_raw_bits'], 'exact action reuse; never nearest-action outcome')
        require(all(row[k] == reference[k] for k in ('root_slot', 'condition', 'arm', 'task_id', 'root_state', 'goal')),
                'same task root goal and arm')
        require(all(c[k] == row[k] for k in ('root_slot', 'condition', 'arm', 'task_id')), 'factorial task identity')
        require(bits(plans['control'][i]['commands'])[:80] == c['prefix_raw_bits'],
                'literal original common prefix, not merely coincident midpoint')
        mid = [[f for f in r['checkpoints'] if f['step'] == 5] for r in (reference, row)]
        require(len(mid[0]) == 1 and mid[0] == mid[1], 'same actual mid snapshot and PNG')
        m = metrics(row)
        require(m['native_success'] is row['native_success'], 'independent native termination metric')
        rows.append(dict(branch=i, anchor=c['anchor'], guard=c['guard'], source=source,
            task_id=c['task_id'], root_slot=c['root_slot'], condition=c['condition'],
            action_bits=c['full10_raw_bits'], model_Q=c['selection']['model_Q'], metrics=m))
    contrasts = []
    for i in range(8):
        z0, z1, a0, a1 = rows[i*4:i*4+4]
        contrasts.append(dict(branch=i, root_slot=z0['root_slot'], condition=z0['condition'],
            anchor_no_guard={k:a0['metrics'][k]-z0['metrics'][k] for k in AXES},
            anchor_guard={k:a1['metrics'][k]-z1['metrics'][k] for k in AXES},
            guard_zero={k:z1['metrics'][k]-z0['metrics'][k] for k in AXES},
            guard_incumbent={k:a1['metrics'][k]-a0['metrics'][k] for k in AXES},
            interaction={k:(a1['metrics'][k]-a0['metrics'][k])-(z1['metrics'][k]-z0['metrics'][k]) for k in AXES}))
    counts = {contrast:{k:dict(better=sum(r[contrast][k] < 0 for r in contrasts),
                              worse=sum(r[contrast][k] > 0 for r in contrasts),
                              tie=sum(r[contrast][k] == 0 for r in contrasts)) for k in AXES}
              for contrast in ('anchor_no_guard', 'anchor_guard', 'guard_zero', 'guard_incumbent')}
    return dict(schema=NAME, cells=rows, contrasts=contrasts, counts=counts, known_roots=2,
        full_factorial_cells=32, distinct_executed_cohorts=3,
        success={a+'_'+str(g):sum(r['metrics']['native_success'] for r in rows if r['anchor']==a and r['guard']==g)
                 for a in ('zero', 'incumbent') for g in (False, True)},
        P1_admission=False, P2_admission=False, independent_confirmation=False, equal_FLOPs=False,
        full_upstream_rebuilt_here=False, new_neural_calls=0, new_physics=0,
        interpretation='Known-root action intervention only; 8 branches are not 8 independent roots. All error axes lower is better.')


def dispatch_plans(path, report):
    journal = path.parent/'journal.jsonl'
    require(sha(journal) == report['journal_sha256'], 'journal bound by pinned actual report')
    with journal.open() as f:
        first = json.loads(f.readline())
    require(first['kind'] == 'run_started', 'literal original dispatch event')
    return first['plans']


def build(zero_report, zero_sha, zero_receipt, receipt_sha):
    bound = {k:read(ROOT/'results'/p, h) for k,(p,h) in PINS.items()}
    z = read(zero_report, zero_sha); strict = read(zero_receipt, receipt_sha)
    require(strict['schema'] == 'agent_p2_safeguard_factorial_physical_20261028_v1.consume'
            and strict['status'] == 'engineering_verified_science_blocked'
            and strict['fixed_denominator'] == 8 and len(strict['rows']) == 8
            and strict['physical_report_sha256'] == zero_sha and strict['P1_admission'] is strict['P2_admission'] is False,
            'external pinned zero physical strict authority, no scientific admission')
    require(bound['incumbent_strict']['physical_report_sha256'] == PINS['incumbent'][1], 'incumbent strict dependency')
    reports = {k:bound[k] for k in ('control', 'incumbent')}; reports['zero'] = z
    paths = {k:ROOT/'results'/PINS[k][0] for k in ('control', 'incumbent')}; paths['zero'] = Path(zero_report)
    require(all(r['errors'] == [] and r['inputs_final_verified'] is True for r in reports.values()), 'clean native cohorts')
    require(dispatch_plans(paths['zero'], z) == strict['plans'], 'zero strict literal dispatch authority')
    result = assemble(bound['cells']['cells'], {k:r['rows'] for k,r in reports.items()},
                      {k:dispatch_plans(paths[k], r) for k,r in reports.items()})
    result['authority'] = dict(pins=PINS, zero_report=str(zero_report), zero_report_sha256=zero_sha,
        zero_receipt=str(zero_receipt), zero_receipt_sha256=receipt_sha,
        reused_pinned_strict_receipts=True, independent_native_state_arithmetic=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('zero-report', 'zero-sha', 'zero-receipt', 'receipt-sha', 'output'):
        p.add_argument('--'+key, required=True)
    a = p.parse_args(); path = Path(a.output).absolute()
    require(path.parent == ROOT/'results' and path.name.startswith(NAME+'_') and not path.is_symlink(), 'exclusive own output')
    result = build(a.zero_report, a.zero_sha, a.zero_receipt, a.receipt_sha)
    with path.open('xb') as f:
        f.write(json.dumps(result, sort_keys=True, allow_nan=False).encode()+b'\n')
    print(json.dumps(dict(output=str(path), sha256=sha(path), counts=result['counts'], success=result['success'])))


if __name__ == '__main__':
    main()
