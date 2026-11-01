"""Independent scalar explanation of the fixed four saved R groups; no producer imports."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = 'main_p3_R_gate_decomposition_20261029_v1'
AUTH = {
    'prepared': ('agent_p3_r_native_group_cuda_20261029_v1_prepare_MAIN01.json', 'c254633fd946f9207c91c13d1961a0bb9dd86c24357a7ca438c1b336ca42b75c'),
    'report': ('agent_p3_r_native_group_cuda_20261029_v1_cuda_MAIN01/report.json', '2c9218f6dfbcf4d24f9a93c9f0ef9dfad2a97f7496adce83bbe0e73cbe8dc354'),
    'receipt': ('agent_p3_r_native_group_cuda_20261029_v1_consume_MAIN01.json', 'c7cedef24438bc6ca95ff2b5c6ec813feb9caee5edc8b47346394d66d8be521a'),
}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def encode(x):
    return (json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def close(a, b):
    need(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12), 'independent scalar mismatch')


def vector(v, n):
    need(type(v) is list and len(v) == n and all(type(x) in (int, float) and math.isfinite(x) for x in v), 'finite scalar vector')
    return v


def mse(x, y):
    return math.fsum((a-b)**2 for a, b in zip(vector(x, 192), vector(y, 192), strict=True))/192


def scalar(row, goal):
    immediate = mse(row['postcontext']['latent_history'][-1], goal)
    nominal = row['signal']['nominal']
    nominal_cost = mse(nominal, goal)
    j, u = row['signal']['jacobian'], vector(row['solve']['params'], 10)
    need(len(j) == 192, 'full postcontext Jacobian')
    residual = [g-n-math.fsum(a*b for a, b in zip(vector(r, 10), u, strict=True))
                for g, n, r in zip(goal, nominal, j, strict=True)]
    linear = math.fsum(x*x for x in residual)/192
    penalty = .002*math.fsum(x*x for x in u)/192
    reach = linear+penalty
    close(immediate, row['immediate'])
    close(linear, row['solve']['linear_terminal_cost'])
    close(penalty, row['solve']['ridge_penalty'])
    close(reach, row['solve']['own_reach_value'])
    close(reach, row['reach'])
    return dict(immediate=immediate, nominal_cost=nominal_cost, reach=reach,
                nominal_drift=nominal_cost-immediate, bounded_linear_gain=nominal_cost-reach,
                linear_terminal_cost=linear, ridge_penalty=penalty)


def explain(p, r, receipt):
    need(receipt['status'] == 'all4_native_R_groups_verified_not_admitted' and receipt['denominator'] == 4, 'complete saved authority')
    need(r['status'] == 'four_groups_complete_not_admitted' and r['errors'] == [], 'complete native run')
    need([(x['root'], x['horizon']) for x in r['rows']] == [(0, 1), (0, 3), (1, 1), (1, 3)], 'all four fixed groups')
    out = []
    for row in r['rows']:
        task = p['tasks'][row['root']]
        need(task['projection'] == [[float(i == j) for j in range(192)] for i in range(192)], 'identity Q only')
        group = row['result']['R_groups'][0]
        zero = scalar(group['control'], task['goal'])
        need([x['index'] for x in group['rows']] == list(range(8)), 'all eight candidates')
        candidates = []
        for c in group['rows']:
            s = scalar(c, task['goal'])
            delta = {k: s[k]-zero[k] for k in ('immediate', 'nominal_cost', 'reach', 'nominal_drift', 'bounded_linear_gain')}
            # reach delta = immediate delta + autonomous drift delta - control gain delta.
            close(delta['reach'], delta['immediate']+delta['nominal_drift']-delta['bounded_linear_gain'])
            nongreedy = c['immediate'] > group['control']['immediate']
            better_reach = c['reach'] < group['control']['reach']
            # Tolerances verify arithmetic only; never change the original exact gate.
            need(c['eligible'] is (nongreedy and better_reach), 'original qualification')
            candidates.append(dict(index=c['index'], scalar=s, delta=delta,
                                   nongreedy=nongreedy, better_reach=better_reach, eligible=c['eligible']))
        need(group['eligible_indices'] == [x['index'] for x in candidates if x['eligible']], 'entire eligibility list')
        out.append(dict(root=row['root'], horizon=row['horizon'], zero=zero, candidates=candidates,
                        counts={k: sum(x[k] for x in candidates) for k in ('nongreedy', 'better_reach', 'eligible')}))
    return dict(schema=NAME+'.analysis', groups=out, groups_count=4, candidates_count=32,
                interpretation='fixed_known2_bank_only; residual_and_control_gain_are_separate; not_a_theory_refutation',
                new_neural_calls=0, new_physics_calls=0, upstream_fullstrict_rebuilt_here=False,
                P1_admission=False, P3_admission=False, freshness=False)


def analyze():
    docs, refs = {}, {}
    for k, (name, h) in AUTH.items():
        path = ROOT/'results'/name
        need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 256*1024**2 and sha(path) == h, 'fixed saved input '+k)
        docs[k] = json.loads(path.read_bytes())
        refs[k] = dict(path=str(path), sha256=h)
    need(docs['receipt']['report_sha256'] == AUTH['report'][1] and docs['receipt']['prepare_sha256'] == AUTH['prepared'][1], 'receipt SHA join')
    result = explain(docs['prepared'], docs['report'], docs['receipt'])
    result['authorities'] = refs
    result['own_sources'] = {str(p): sha(p) for p in (Path(__file__), ROOT/'tests'/('test_'+NAME+'.py'))}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('analyze', 'verify'))
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    need(a.output.parent.resolve() == ROOT/'results' and a.output.name.startswith(NAME+'_') and not a.output.is_symlink(), 'exclusive result scope')
    result = analyze()
    if a.command == 'verify':
        need(a.output.read_bytes() == encode(result), 'entire deterministic result')
    else:
        with a.output.open('xb') as f:
            f.write(encode(result)); f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(verified=True, path=str(a.output), sha256=sha(a.output))))


if __name__ == '__main__':
    main()
