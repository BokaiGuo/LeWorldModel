"""Independent saved scalar Taylor decomposition; no neural/model imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path

NAME = 'main_p3_goal_bank_curvature_20261029_v1'
REPORT_SHA = '9bedec46c5c0e74b73c145935280b2efe400f617f8bbe7b72205fedb60a20d4f'
PRODUCER = 'agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def encode(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def vector(x, n):
    need(type(x) is list and len(x) == n and all(type(v) in (int, float) and math.isfinite(v) for v in x), 'finite numeric vector')
    return x


def dot(a, b):
    need(len(a) == len(b), 'dot dimensions')
    return math.fsum(x*y for x, y in zip(a, b, strict=True))


def close(a, b):
    # Diagnostic arithmetic bound, NOT an eligibility tolerance.
    need(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-11*max(1., abs(a), abs(b)), 'scalar reconstruction')


def decompose(goal, nominal, jacobian, actions, actual):
    d = len(goal)
    need(d > 0, 'nonempty task')
    goal, nominal, actual = [vector(v, d) for v in (goal, nominal, actual)]
    need(type(jacobian) is list and len(jacobian) == d, 'Jacobian rows')
    u = vector(actions, len(actions))
    v = [dot(vector(row, len(u)), u) for row in jacobian]
    r = [n-g for n, g in zip(nominal, goal, strict=True)]
    e = [z-n-delta for z, n, delta in zip(actual, nominal, v, strict=True)]
    first, quadratic = 2*dot(r, v)/d, dot(v, v)/d
    nonlinear_cross = 2*dot([a+b for a, b in zip(r, v, strict=True)], e)/d
    nonlinear_square = dot(e, e)/d
    nominal_cost = dot(r, r)/d
    actual_r = [z-g for z, g in zip(actual, goal, strict=True)]
    actual_cost = dot(actual_r, actual_r)/d
    delta = actual_cost-nominal_cost
    total = math.fsum((first, quadratic, nonlinear_cross, nonlinear_square))
    close(delta, total)
    return dict(nominal_cost=nominal_cost, actual_cost=actual_cost, delta=delta,
                first_order=first, quadratic=quadratic, nonlinear_cross=nonlinear_cross,
                nonlinear_square=nonlinear_square, nonlinear_total=nonlinear_cross+nonlinear_square,
                reconstructed_delta=total, reconstruction_error=delta-total)


def analyze(report):
    need(report['schema'] == PRODUCER+'.report' and report['errors'] == [] and
         report['status'] == 'four_slots_settled_not_admitted', 'complete actual report')
    need([(r['root'], r['horizon']) for r in report['rows']] == [(0, 1), (0, 3), (1, 1), (1, 3)], 'fixed four groups')
    groups = []
    for row in report['rows']:
        result = row['result']; group = result['R_groups'][0]; bank = group['bank']
        need(result['errors'] == [] and bank['status'] == 'complete', 'all eight finite candidates')
        goal = bank['local_task']['goal']; response = bank['native_response']
        nominal, jacobian = response['nominal'], response['jacobian']
        need(nominal == group['control']['postcontext']['latent_history'][-1], 'same-call zero edge exact nominal')
        need(len(goal) == 192 and bank['local_task']['projection'] == [[int(i == j) for j in range(192)] for i in range(192)], 'identity192 task')
        need([r['index'] for r in group['rows']] == list(range(8)), 'full candidate denominator')
        rows = []
        for candidate, generated in zip(group['rows'], bank['candidates'], strict=True):
            need(candidate['actions'] == generated['actions'], 'actual evaluated bank actions')
            values = decompose(goal, nominal, jacobian, [v for packet in candidate['actions'] for v in packet],
                               candidate['postcontext']['latent_history'][-1])
            close(values['actual_cost'], candidate['immediate'])
            close(values['nominal_cost'], group['control']['immediate'])
            close(values['first_order'], -2*generated['b_dot_cast32_action']/192)
            nongreedy = candidate['immediate'] > group['control']['immediate']
            better = candidate['reach'] < group['control']['reach']
            need(candidate['eligible'] is (nongreedy and better), 'saved exact gate, no epsilon')
            rows.append(dict(index=candidate['index'], direction=generated['direction'], sign=generated['sign'],
                             radius=generated['radius'], nongreedy=nongreedy, better_reach=better,
                             eligible=candidate['eligible'], reach_delta=candidate['reach']-group['control']['reach'], **values))
        groups.append(dict(root=row['root'], horizon=row['horizon'], candidates=rows,
                           nongreedy=sum(r['nongreedy'] for r in rows), better_reach=sum(r['better_reach'] for r in rows),
                           eligible=sum(r['eligible'] for r in rows),
                           first_order_vs_actual_sign_reversals=sum(r['first_order']*r['delta'] < 0 for r in rows)))
    return dict(schema=NAME+'.diagnostic', groups=groups, denominator=32, neural_calls=0, physics_calls=0,
                independent_scalar_math=True, source_IO_authentication='inherited full consumer; not rebuilt here',
                eligibility_tolerance_changed=False, general_impossibility=False, scientific_admission=False)


def build(report_path, receipt_path, receipt_sha):
    p, c = Path(report_path).resolve(), Path(receipt_path).resolve()
    need(p.stat().st_size <= 512*1024**2 and c.stat().st_size <= 128*1024**2, 'bounded inputs')
    need(sha(p) == REPORT_SHA and sha(c) == receipt_sha, 'explicit real input SHAs')
    receipt = json.loads(c.read_bytes())
    need(receipt['schema'] == PRODUCER+'.consume' and receipt['report_sha256'] == REPORT_SHA and receipt['denominator'] == 4,
         'actual complete consumer authority')
    out = analyze(json.loads(p.read_bytes()))
    out['inputs'] = dict(report=str(p), report_sha256=REPORT_SHA, receipt=str(c), receipt_sha256=receipt_sha)
    out['sources'] = {str(q): sha(q) for q in (Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'))}
    need(sha(p) == REPORT_SHA and sha(c) == receipt_sha, 'stable inputs')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('analyze', 'verify'))
    for key in ('report', 'receipt', 'receipt-sha', 'output'):
        ap.add_argument('--'+key, required=True)
    args = ap.parse_args(); out = build(args.report, args.receipt, args.receipt_sha)
    path = Path(args.output).absolute()
    need(path.parent == ROOT/'results' and path.name.startswith(NAME+'_') and not path.is_symlink(), 'exclusive own result path')
    if args.command == 'analyze':
        import os
        with path.open('xb') as f:
            f.write(encode(out)); f.flush(); os.fsync(f.fileno())
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    else:
        need(path.stat().st_size < 4*1024**2 and path.read_bytes() == encode(out), 'entire saved reconstruction')
    print(json.dumps(dict(path=str(path), sha256=sha(path), groups=[{k:g[k] for k in ('root','horizon','nongreedy','better_reach','eligible','first_order_vs_actual_sign_reversals')} for g in out['groups']])))


if __name__ == '__main__':
    main()
