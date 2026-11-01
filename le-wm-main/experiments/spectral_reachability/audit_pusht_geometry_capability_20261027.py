"""Independent saved-receipt arithmetic audit; no neural or simulator calls."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.spectral_reachability.p3_matched_physical_20261027_v1 import digest, write


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        and np.allclose(a, b, rtol=1e-9, atol=1e-10), message)


def feature(state):
    s = np.asarray(state, dtype=float)
    return np.array([s[2], s[3], 20*np.cos(s[4]), 20*np.sin(s[4])])/512


def fit_check(k, r, fit):
    """Check independent normal equations, ball KKT and objective, not solver reuse."""
    k, r, u = np.asarray(k), np.asarray(r), np.asarray(fit['params'])
    lam, cap, mu = .002, np.sqrt(5)*.75, fit['ball_multiplier']
    require(mu >= 0 and np.linalg.norm(u) <= cap+1e-9, 'fit ball constraint')
    close(k.T@(k@u-r)+(lam+mu)*u, np.zeros(10), 'fit KKT stationarity')
    require(abs(mu*(np.linalg.norm(u)**2-cap**2)) < 1e-8, 'fit complementarity')
    cost = ((r-k@u)@(r-k@u)+lam*(u@u))/len(r)
    close(cost, fit['own_reach_value'], 'fit objective')
    return float(cost)


def audit(report, protocol, compilation=None):
    require(report['status'] == 'diagnostic_recorded' and not report['failures']
        and bool(report['gates']) and all(v is True for v in report['gates'].values()),
        'complete actual diagnostic required')
    plans = {(p['root_sha256'], p['method']):p for p in protocol['plans']}
    rows = {(r['root_sha256'], r['method']):r for r in report['rows']}
    slots = {(s['root_sha256'], s['method'], s['kind']):s for s in report['endpoint_receipts']}
    enc = {(e['root_sha256'], e['method'], e['kind']):e for e in report['encodings']}
    require(len(plans) == len(protocol['plans']) == len(rows) == len(report['rows']) == 123
        and set(plans) == set(rows), 'fixed complete model-qualified census')
    expected = {(*key, kind) for key in plans for kind in ('prefix', 'terminal')}
    require(set(slots) == set(enc) == expected and len(slots) == len(enc) == 246
        and len(report['endpoint_receipts']) == len(report['encodings']) == 246, 'endpoint census')
    for key, row in rows.items():
        require(row['status'] == 'completed' and row['completed_commands'] == 10
            and len(row['trace']) == 10 and row['full_horizon_completed'], 'complete horizon')
        close([e['command'] for e in row['trace']], plans[key]['commands'], 'commands differ')
        for kind, step in (('prefix', 5), ('terminal', 10)):
            slot, e = slots[*key, kind], enc[*key, kind]
            require(slot['status'] == 'observed' and slot['primitive_step'] == step
                and e['status'] == 'completed' and e['comparable']
                and e['attempted_encode_samples'] == 1, 'exact encoded endpoint required')
            close(slot['state'], row['trace'][step-1]['state'], 'state binding')
            close(slot['phi'], feature(slot['state']), 'physical feature binding')
    responses = report['analysis']['responses']
    metrics = {} if compilation is None else {r['root_sha256']:
        (np.asarray(r['compiled']['goal_vectors'][0]),
         np.asarray(r['compiled']['scoring']['metric']['projection'])) for r in compilation['rows']}
    require(len(responses) == 6 and len({(r['role'], r['space']) for r in responses}) == 6,
        'all role/space fits retained')
    findings = []
    for response in responses:
        root, role, space = [response[k] for k in ('root_sha256','role','space')]
        require(response['status'] == 'completed', 'all response groups required')
        def vector(method):
            return np.asarray(enc[root, method, 'terminal']['vector'] if space == 'real_Q'
                else slots[root, method, 'terminal']['phi'])
        base, common_base = vector(role+'_zero'), vector('zero_zero')
        close(base, response['base_terminal'], 'role baseline binding')
        close(common_base, response['zero_role_terminal'], 'common baseline binding')
        target = feature(rows[root, role+'_zero']['context_identity']['goal']) if space == 'physical_phi' else (
            metrics[root][0] if compilation is not None else np.asarray(response['target']))
        projection = np.eye(4) if space == 'physical_phi' else (metrics[root][1] if compilation is not None else None)
        close(target, response['target'], 'original target binding')
        close(target-base, response['own_raw_residual'], 'own residual binding')
        close(target-common_base, response['common_raw_residual'], 'common residual binding')
        if projection is not None:
            close(projection@(target-base), response['own_residual'], 'projected own residual')
            close(projection@(target-common_base), response['common_residual'], 'projected common residual')
        require([g['epsilon'] for g in response['grids']] == [.05, .1], 'fixed epsilons')
        norms = []
        for grid_index, grid in enumerate(response['grids']):
            eps = grid['epsilon']
            raw = np.stack([(vector(f'{role}_e{grid_index}_a{axis}_plus')-
                vector(f'{role}_e{grid_index}_a{axis}_minus'))/(2*eps) for axis in range(10)], axis=1)
            close(raw, grid['raw_jacobian'], 'raw finite difference binding')
            k = np.asarray(grid['response'])
            if projection is not None:
                close(k, projection@raw, 'original projection binding')
            fit_check(k, np.asarray(response['own_residual']), grid['own_fit'])
            fit_check(k, np.asarray(response['common_raw_residual']) if space == 'physical_phi'
                else np.asarray(response['common_residual']), grid['common_zero_role_fit'])
            norms.append(float(np.linalg.norm(raw)))
        findings.append(dict(role=role, space=space, raw_response_norms=norms,
            epsilon_sensitivity=response['epsilon_sensitivity']))
    steps = sum(r['physical_env_step_returns'] for r in rows.values())
    require(steps == report['physical_env_step_returns'] == report['physical_env_step_attempts'] == 1230,
        'actual execution accounting')
    require(sum(e['attempted_encode_samples'] for e in enc.values()) == report['attempted_encode_samples'] == 246,
        'actual encoder accounting')
    require(report['new_predict_calls'] == 0 and report['P1_admission'] is False, 'diagnostic scope')
    return dict(status='saved_receipt_arithmetic_verified', findings=findings,
        actual_steps=steps, actual_cuda_encoding_samples=246, P1_admission=False,
        interpretation='Known-root small-probe response only. RGB response is not object control. '
            'Own-residual and common-residual fits answer different questions; cross-space magnitudes '
            'are not comparable. Neither global theory failure nor encoder/controller repair is established.')


def run(input_path, output):
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    input_path, output = Path(input_path).resolve(), Path(output).resolve()
    require(not output.exists(), 'exclusive output required')
    report = json.loads(input_path.read_text())
    protocol_path = input_path.parent/'protocol.json'
    require(digest(protocol_path) == report['protocol_sha256'], 'protocol changed')
    protocol = json.loads(protocol_path.read_text())
    for seal in report['source_seals']:
        verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    for path, sha in protocol['bindings'].items():
        require(digest(path) == sha, 'bound input changed')
    planning_path = Path(protocol['planning_report'])
    planning = json.loads(planning_path.read_text())
    compilation_path = Path(planning['protocol']['compilation'])
    require(str(compilation_path) in protocol['bindings'], 'compilation must be bound')
    compilation = json.loads(compilation_path.read_text())
    result = audit(report, protocol, compilation)
    for row in report['rows']:
        require(digest(row['journal_path']) == row['journal_sha256'], 'journal changed')
    for slot in report['endpoint_receipts']:
        require(digest(slot['rgb_path']) == slot['rgb_sha256'], 'endpoint RGB changed')
    output.mkdir(exist_ok=False)
    result['inputs'] = {str(p):digest(p) for p in (input_path, protocol_path)}
    result['source_seal'] = create_seal(files=[Path(__file__), Path(__file__).resolve().parents[2]/
        'tests/test_audit_pusht_geometry_capability_20261027.py'], args_json=json.dumps(result['inputs']),
        output_dir=output/'source_seal')
    write(output/'report.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(**vars(parser.parse_args()))
