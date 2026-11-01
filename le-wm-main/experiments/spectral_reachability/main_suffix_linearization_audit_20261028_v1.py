"""Separate saved local-solve objective from nonlinear deployment objective."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p2_contextual_midfeedback_recompile_20261028_v1 as source

ROOT = source.ROOT
NAME = 'main_suffix_linearization_audit_20261028_v1'
FOLDER = ROOT/'results/agent_p2_contextual_midfeedback_recompile_20261028_v1_cuda_MAIN01'
REPORT_SHA = '8146692ccc9a3c2c6c09ff3c99419274e2e8a8ff86bb3ac0465af01dc6401eea'
STRICT = ROOT/'results/agent_p2_contextual_midfeedback_recompile_20261028_v1_compile_consume_MAIN01.json'
STRICT_SHA = 'f31377e78981fb4a84c04fb4a790dcbe1b3a23e00c7202a7d73bf9311c043094'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def terms(nominal, jac, params, actual, goal, damping=.002):
    n, j, p, z, g = [np.asarray(v, np.float64) for v in (nominal, jac, params, actual, goal)]
    require(n.shape == z.shape == g.shape == (192,) and j.shape == (192, 10) and p.shape == (10,) and
            all(np.isfinite(v).all() for v in (n, j, p, z, g)), 'native dimensions and finite arrays')
    step = j@p; r = n-g; e = z-(n+step)
    mse = lambda v: float(np.mean(v*v))
    linear_delta = 2*float(np.mean(r*step))+mse(step)
    cross = 2*float(np.mean((r+step)*e)); curvature_square = mse(e)
    nonlinear_delta = mse(z-g)-mse(r)
    residual = nonlinear_delta-linear_delta-cross-curvature_square
    require(abs(residual) <= 64*np.finfo(float).eps*max(1., abs(nonlinear_delta), abs(linear_delta), abs(cross), curvature_square),
            'float64 expansion reconstruction; numerical audit tolerance is not a scientific gate')
    return dict(nominal_Q=mse(r), local_linear_Q=mse(r+step), nonlinear_new_Q=mse(z-g),
        linear_delta=linear_delta, nonlinear_delta=nonlinear_delta,
        nonlinear_remainder_cross=cross, nonlinear_remainder_square=curvature_square,
        nonlinear_remainder_MSE=mse(e), expansion_residual=residual,
        ridge_penalty_per_dimension=float(damping*(p@p)/192),
        local_regularized_delta=linear_delta+float(damping*(p@p)/192))


def build():
    require(source.sha(FOLDER/'report.json') == REPORT_SHA and source.sha(STRICT) == STRICT_SHA, 'pinned actual compile and completed strict')
    report = source.physical.read(FOLDER/'report.json'); strict = source.physical.read(STRICT)
    require(strict['status'] == 'compiled_engineering_only' and len(strict['plans']) == len(report['rows']) == 8, 'all8 completed compile')
    prepared = source.physical.read(report['prepared'])
    require(source.sha(report['prepared']) == report['prepare_sha256'], 'actual prepared context/goal identity')
    rows = []
    for i, (b, row) in enumerate(zip(prepared['branches'], report['rows'], strict=True)):
        require(row['index'] == i and np.array_equal(b['q'], np.eye(192)), 'same8 identity goal metric')
        load = lambda k: source.load_npy(FOLDER, row['arrays'][k])
        n, j, p = load('nominal'), load('jacobian'), load('new_logits')
        z = np.asarray(row['evaluations'][1]['context']['latent_history'][-1], np.float32)
        result = terms(n, j, p, z, b['goal'])
        require(result['nonlinear_new_Q'] == row['evaluations'][1]['cost'], 'independent same native nonlinear Q')
        old_q = row['evaluations'][0]['cost']
        mapped = np.tanh(np.asarray(p, np.float32)).reshape(5, 2)
        require(mapped.tobytes() == np.asarray(row['new_suffix_commands'], np.float32).tobytes(), 'actual rounded logit chart')
        rows.append(dict(index=i, **result, old_actual_mid_Q=old_q,
            new_minus_old_Q=result['nonlinear_new_Q']-old_q,
            rounded_logit_max_abs=float(np.max(np.abs(np.asarray(p, np.float32).astype(np.float64)-p))),
            actual_float32_chart_linear_Q=terms(n,j,np.asarray(p,np.float32),z,b['goal'])['local_linear_Q']))
    return dict(schema=NAME, rows=rows, fixed_plans=8, known_roots=2, fresh_roots=0,
        linear_regularized_improved=sum(r['local_regularized_delta'] < 0 for r in rows),
        nonlinear_improved_over_zero=sum(r['nonlinear_delta'] < 0 for r in rows),
        nonlinear_worse_than_incumbent=sum(r['new_minus_old_Q'] > 0 for r in rows),
        authority=dict(report_sha256=REPORT_SHA, completed_strict_sha256=STRICT_SHA, raw_NPY_recomputed=True,
            full_upstream_rebuilt_here=False), new_neural_calls=0, new_physics=0,
        P1_admission=False, P2_admission=False,
        interpretation='Local linear objective, nonlinear model objective, and physical outcomes are different tests; no outcomes used for a new selector.')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True)
    a = p.parse_args(); output = Path(a.output).absolute()
    require(output.resolve() == output and output.parent == ROOT/'results' and output.name.startswith(NAME+'_'), 'own new output')
    result = build()
    with output.open('xb') as f:
        f.write(json.dumps(result, sort_keys=True, allow_nan=False).encode()+b'\n')
    print(json.dumps(dict(output=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        linear_improved=result['linear_regularized_improved'], nonlinear_improved=result['nonlinear_improved_over_zero'],
        worse_than_incumbent=result['nonlinear_worse_than_incumbent'])))


if __name__ == '__main__':
    main()
