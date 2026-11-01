"""Saved-only constrained stationarity diagnostic, not a stopping threshold."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/p1_observed_direct_cuda_20261027_v2_cuda_main01'
REPORT_SHA = '954af696d88eecebdcdc7345de702809517491967263f395a8f2819753d889d5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)


def project(x, radius):
    """Exact real-algebra projection, NOT the producer's float32 rounding."""
    x = np.asarray(x, dtype=float)
    if x.shape != (2, 10) or not np.isfinite(x).all() or not np.isfinite(radius) or radius <= 0:
        raise ValueError('finite two independent packet balls')
    return x * np.minimum(1., radius / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-300))


def analyze(task, report):
    goal, q = np.asarray(task['goal']), np.asarray(task['projection'])
    trajectory = report['direct_runs'][0]
    if len(trajectory) != 9 or [s['iteration'] for s in trajectory] != list(range(9)):
        raise ValueError('all original B8 steps required')
    rows = []
    for before, step in zip(trajectory, trajectory[1:]):
        d = step['derivative']
        if d['at_logits'] != before['incumbent']['logits'] or d['at_terminal'] != before['incumbent']['contexts'][-1]['latent_history'][-1]:
            raise ValueError('derivative point must be actual preceding incumbent')
        x, j = np.asarray(d['at_logits']), np.asarray(d['raw_terminal_jacobian'])
        residual = q @ (np.asarray(d['at_terminal']) - goal)
        gradient = 2 * (q @ j).T @ residual / len(goal)
        np.testing.assert_allclose(gradient, d['objective_gradient'], rtol=1e-12, atol=1e-14)
        proposal = np.asarray(step['proposal']['logits'])
        displacement = proposal - x
        cost_before, cost_proposal = before['incumbent']['cost'], step['proposal']['cost']
        if step['accepted'] != (cost_proposal < cost_before):
            raise ValueError('strict original acceptance rule')
        expected = step['proposal'] if step['accepted'] else before['incumbent']
        if step['incumbent'] != expected:
            raise ValueError('wrong incumbent after acceptance')
        mapping = x - project(x - gradient.reshape(2, 10), task['packet_logit_cap'])
        rows.append(dict(iteration=step['iteration'], cost_before=cost_before,
            proposal_cost=cost_proposal, cost_after=step['incumbent']['cost'], accepted=step['accepted'],
            gradient_norm=float(np.linalg.norm(gradient)),
            projected_gradient_mapping_norm_unit_step=float(np.linalg.norm(mapping)),
            proposal_displacement_norm=float(np.linalg.norm(displacement)),
            gradient_dot_proposal_displacement=float(gradient @ displacement.reshape(20)),
            packet_logit_norms=np.linalg.norm(x, axis=1).tolist()))
    return dict(method=report['configuration']['method'], rows=rows,
        B4_cost=trajectory[4]['incumbent']['cost'], B8_cost=trajectory[8]['incumbent']['cost'],
        optimization_success_not_physical_success=True, stationarity_threshold=None,
        caveat='Nonzero saved gradient is conditional on saved AD. No new FD or neural prediction; no scientific plateau threshold.')


def run(output):
    path = RUN / 'report.json'
    if sha(path) != REPORT_SHA:
        raise ValueError('fixed actual CUDA report')
    report = read(path)
    if report['status'] != 'observed_direct_calibration_completed' or report['errors'] or not report['model_unchanged']:
        raise ValueError('complete frozen CUDA evidence')
    bindings = {str(path): REPORT_SHA}
    tasks = {}
    for i in range(2):
        p = RUN / f'task_{i}.json.gz'
        encoded = read(p)
        tasks[encoded['root_sha256']] = encoded['task']
        bindings[str(p)] = sha(p)
    rows = []
    for arm in report['direct']:
        for field in ('report', 'journal'):
            p = Path(arm[field + '_path'])
            if sha(p) != arm[field + '_sha256']:
                raise ValueError('original actual arm hash')
            bindings[str(p)] = arm[field + '_sha256']
        rows.append(dict(root_sha256=arm['root_sha256'], **analyze(tasks[arm['root_sha256']], read(arm['report_path']))))
    result = dict(schema='p1_direct_stagnation_saved_main_20261027_v1', rows=rows,
        bindings=bindings, new_neural_calls=0, new_physical_steps=0, P1_admission=False,
        scientific_stationarity_certificate=False, source_sha256=sha(__file__))
    for p, expected in bindings.items():
        if sha(p) != expected:
            raise ValueError('saved evidence changed during read')
    raw = (json.dumps(result, sort_keys=True, allow_nan=False) + '\n').encode()
    output = Path(output)
    if len(raw) > 128 * 1024 or shutil.disk_usage(output.parent).free < 200 * 1024**2 + len(raw):
        raise ValueError('small output and unchanged reserve')
    with output.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    r = run(p.parse_args().output)
    print(json.dumps([dict(root=x['root_sha256'][:8], method=x['method'], B4=x['B4_cost'], B8=x['B8_cost'],
        last_projected_gradient=x['rows'][-1]['projected_gradient_mapping_norm_unit_step'],
        last_g_dot_step=x['rows'][-1]['gradient_dot_proposal_displacement']) for x in r['rows']]))
