"""Saved known-root FD scalar decomposition; no neural/physical calls or repair fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

NAME = 'main_observed_fd_scalar_cause_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
PHYSICAL = ROOT/'results/pusht_observed_suffix_physical_fd4_v1_20261026_cuda/report.json'
NEURAL = ROOT/'results/pusht_observed_suffix_neural_fd4_v1_20261026_cuda/report.json'
OBSERVED = ROOT/'results/pusht_observed_oneshot_suffix_control4_v1_20261026_cuda/report.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(value):
    def typed(x):
        if isinstance(x, list):
            for y in x:
                typed(y)
        else:
            require(type(x) in (int, float), 'literal numeric arrays, not bool')
    typed(value)
    a = np.asarray(value, float)
    require(np.isfinite(a).all(), 'finite numeric arrays')
    return a


def scalar_cause(pm, pp, am, ap, goal, projection, span):
    """Exact quadratic secant identity up to explicit float64 arithmetic residual.

    midpoint differences and response differences are separate AND interacting.
    Neither algebraic deletion nor oracle physical gradients define a repair.
    """
    pm, pp, am, ap, goal, projection = map(array, (pm, pp, am, ap, goal, projection))
    require(goal.ndim == 1 and all(x.shape == goal.shape for x in (pm, pp, am, ap))
            and projection.ndim == 2 and projection.shape[1] == len(goal), 'metric dimensions')
    require(type(span) in (int, float) and np.isfinite(span) and span > 0, 'positive actual parameter span')
    jp, ja = (projection@(pp-pm))/span, (projection@(ap-am))/span
    rp = projection@((pp+pm)/2-goal)
    ra = projection@((ap+am)/2-goal)
    dj, dr = jp-ja, rp-ra
    inner = lambda x, y: float(np.mean(x*y))
    q = lambda x: inner(projection@(x-goal), projection@(x-goal))
    gp, ga = (q(pp)-q(pm))/span, (q(ap)-q(am))/span
    terms = dict(response=2*inner(dj, ra), midpoint_bias=2*inner(ja, dr), interaction=2*inner(dj, dr))
    residual = gp-ga-sum(terms.values())
    require(abs(residual) <= 1e-9*(1+abs(gp)+abs(ga)+sum(map(abs, terms.values()))), 'quadratic secant identity')
    return dict(model_scalar_fd=gp, actual_scalar_fd=ga, strict_direction_reversal=bool(gp*ga < 0),
                scalar_identity_residual=residual, terms=terms, actual_span=span,
                opposing_terms=bool(min(terms.values()) < 0 < max(terms.values())),
                model_secant_norm=float(np.linalg.norm(jp)), actual_secant_norm=float(np.linalg.norm(ja)))


def keyed(rows, key):
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows) and rows, 'nonempty unique identities')
    return result


def read(path):
    return json.loads(Path(path).read_bytes())


def build():
    n, p, o = map(read, (NEURAL, PHYSICAL, OBSERVED))
    require(n['schema'] == 'pusht_observed_suffix_neural_fd.v1'
            and p['schema'] == 'pusht_observed_suffix_physical_fd.v1'
            and o['schema'] == 'pusht_observed_oneshot_suffix_control.v1', 'exact original schemas')
    for report in (n, p, o):
        require(report['status'] == 'completed' and report['P1_admission'] is False
                and report['gates'] and all(v is True for v in report['gates'].values()), 'completed saved diagnostics')
    for path in (PHYSICAL, OBSERVED):
        require(n['protocol']['bindings'][str(path)] == sha(path), 'saved neural report binds physical/observed bytes')
    bindings = {str(x): sha(x) for x in (NEURAL, PHYSICAL, OBSERVED, Path(__file__))}
    nm, pm, om = [keyed(r['rows'], 'root_sha256') for r in (n, p, o)]
    require(len(nm) == 4 and nm.keys() == pm.keys() == om.keys(), 'same four fixed known roots')
    rows = []
    for root in n['rows']:
        key = root['root_sha256']
        nb, pb, ob = [keyed(r['branches'], 'candidate_id') for r in (root, pm[key], om[key])]
        require(len(nb) == 2 and nb.keys() == pb.keys() and nb.keys() <= ob.keys()
                and 'prefix_0' in nb, 'same two frozen FD branches within original all16 observed branches')
        for name, neural in nb.items():
            physical, observed = pb[name], ob[name]
            path = Path(observed['plan_path'])
            require(sha(path) == observed['plan_sha256'] == n['protocol']['bindings'][str(path)], 'original unchanged goal/Q plan')
            bindings[str(path)] = sha(path)
            plan = read(path)
            require(plan['root_sha256'] == key and plan['candidate_id'] == name, 'plan root/candidate identity')
            goal, projection = array(plan['goal']), array(plan['projection'])
            nn, aa = [keyed(b['probes'], 'id') for b in (neural, physical)]
            require(len(nn) == 9 and nn.keys() == aa.keys(), 'all9 probes, none dropped')
            require(np.array_equal(array(neural['saved_nominal']), array(plan['nominal']))
                    and np.array_equal(array(physical['saved_model_zero_nominal']), array(plan['nominal']))
                    and np.array_equal(array(neural['saved_ad_jacobian']), array(plan['jacobian']))
                    and np.array_equal(array(physical['saved_model_jacobian']), array(plan['jacobian'])), 'original numerical anchors')
            for ident, npb in nn.items():
                apb = aa[ident]
                for field in ('id', 'epsilon', 'axis', 'sign', 'params', 'executed_params', 'prefix_rgb_sha256'):
                    require(npb[field] == apb[field], 'same physical/neural probe '+field)
                require(npb['status'] == apb['status'] == 'executed'
                        and npb['prefix_replay_exact'] is apb['prefix_replay_exact'] is True, 'exact prefix replays')
                require(len(array(npb['vector'])) == len(array(apb['vector'])) == len(goal), 'full terminal vectors')
            comparisons = []
            for h in (.01, .05):
                for axis in (0, 1):
                    ids = [f'e{h}.a{axis}.s{sign}' for sign in (-1, 1)]
                    mn, pn = [nn[i] for i in ids]
                    ma, pa = [aa[i] for i in ids]
                    x, y = array(mn['executed_params']), array(pn['executed_params'])
                    span = float(y[axis]-x[axis])
                    require(np.count_nonzero(y-x) == 1 and span == 2*float(np.float32(h)), 'original atanh-parameter span, NOT raw action')
                    item = scalar_cause(mn['vector'], pn['vector'], ma['vector'], pa['vector'],
                                        goal.tolist(), projection.tolist(), span)
                    item.update(epsilon=h, axis=axis)
                    comparisons.append(item)
            rows.append(dict(root_sha256=key, root_index=root['root_index'], candidate_id=name, comparisons=comparisons))
    values = [c for r in rows for c in r['comparisons']]
    require(len(rows) == 8 and len(values) == 32, 'fixed all8 contexts/all32 coordinate-scale comparisons')
    return dict(schema=NAME, rows=rows, bindings=bindings,
                comparisons=32, strict_direction_reversals=sum(c['strict_direction_reversal'] for c in values),
                opposing_term_comparisons=sum(c['opposing_terms'] for c in values),
                max_identity_residual=max(abs(c['scalar_identity_residual']) for c in values),
                new_neural_calls=0, new_physics_calls=0, P1_admission=False, selector_changed=False,
                full_upstream_rebuilt_here=False, fresh_confirmation=False,
                scope='Reanalysis of saved old4-root repeated-packet parameter FD, not current raw10 cohort. '
                      'No fitted correction; decomposition is not unique causal attribution or deployed oracle policy.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).absolute()
    require(output.resolve() == output and output.parent == ROOT/'results'
            and output.name.startswith(NAME+'_') and output.suffix == '.json', 'own exclusive result')
    result = build()
    with output.open('xb') as f:
        f.write(json.dumps(result, sort_keys=True, allow_nan=False).encode()+b'\n')
    print(json.dumps({k: result[k] for k in ('comparisons', 'strict_direction_reversals', 'opposing_term_comparisons', 'max_identity_residual')}))


if __name__ == '__main__':
    main()
