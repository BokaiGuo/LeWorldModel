"""Saved-only audit of all four banks; descriptive, no action release/reselection."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import os
import numpy as np

REPORT_SHA = '2b2e5064f74f8aca712ae0d0415fcbfb64a2c8e6325bdffe24ae4fdc3b373381'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cost(z, goal, q):
    z, goal, q = map(lambda x: np.asarray(x, dtype=np.float64), (z, goal, q))
    require(z.shape == goal.shape == (192,) and q.shape == (192, 192), 'full latent metric shapes')
    require(all(np.isfinite(x).all() for x in (z, goal, q)), 'finite arrays')
    return float(np.mean((q @ (z-goal))**2))


def bank(result, task):
    rows = []
    require(len(result['candidates']) == 10, 'complete candidate bank')
    for i, c in enumerate(result['candidates']):
        require(c['index'] == i and c['completed'] is True and c['independent_model_replay_exact'] is True, 'fixed complete bank')
        stage = cost(c['predicted_prefix'], task['goal'], task['projection'])
        nonlinear = cost(c['predicted_terminal'], task['goal'], task['projection'])
        j = np.asarray(c['jacobian'], dtype=np.float64)
        n = np.asarray(c['nominal_terminal'], dtype=np.float64)
        u = np.asarray(c['suffix_params'], dtype=np.float64)
        q = np.asarray(task['projection'], dtype=np.float64)
        require(j.shape == (192, 10) and n.shape == (192,) and u.shape == (10,), 'saved linearization')
        residual = q @ (np.asarray(task['goal']) - n - j @ u)
        linear = float(residual @ residual / 192)
        penalty = float(.002 * (u @ u) / 192)
        solve = c['suffix_solve']
        np.testing.assert_allclose([stage, nonlinear, linear, penalty],
            [c['stage_cost'], c['nonlinear_terminal_cost'], solve['linear_terminal_cost'], solve['ridge_penalty']],
            rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(c['objective'], stage+linear+penalty, rtol=1e-12, atol=1e-12)
        rows.append(dict(index=i, stage=stage, linear_terminal=linear, nonlinear_terminal=nonlinear,
                         ridge_penalty=penalty, declared_objective=c['objective'],
                         saved_nonlinear_analog=stage+nonlinear+penalty,
                         nonlinear_minus_linear=nonlinear-linear))
    original = min(rows, key=lambda r: (r['declared_objective'], r['index']))
    require(result['winner'] == result['candidates'][original['index']], 'declared winner preserved')
    analog = min(rows, key=lambda r: (r['saved_nonlinear_analog'], r['index']))
    return dict(original_index=original['index'], diagnostic_analog_index=analog['index'],
                diagnostic_ranking_disagrees=original['index'] != analog['index'],
                saved_model_analog_gap=original['saved_nonlinear_analog']-analog['saved_nonlinear_analog'],
                candidates=rows, commands_released=False)


def audit(directory):
    directory = Path(directory)
    require(sha(directory/'report.json') == REPORT_SHA, 'fixed original CUDA report')
    report = json.loads((directory/'report.json').read_bytes())
    require(report['errors'] == [] and report['model_unchanged'] is True, 'complete frozen source')
    def saved(name):
        path = directory/name
        require(sha(path) == report['artifact_sha256'][name], 'saved candidate/task artifact SHA')
        return json.loads(gzip.decompress(path.read_bytes()))
    rows = []
    for i in (1, 3, 5, 7):
        plan = report['rows'][i]
        task = saved(plan['task_file'])
        result = saved(plan['result_file'])
        row = bank(result, task['task'])
        row.update(root_slot=plan['root_slot'], condition=plan['condition'], task_id=plan['task_id'])
        require(task['task_id'] == plan['task_id'], 'task join')
        rows.append(row)
    return dict(status='saved_arithmetic_verified', source_report_sha256=REPORT_SHA,
                banks=4, candidates=40, rows=rows, new_neural_calls=0, new_physics_calls=0,
                P1_admission=False, commands_released=False,
                interpretation='implementation matches declared linear objective; diagnostic nonlinear ranking is not an evaluated fix',
                limitations=['two known calibration roots', 'model scores are not physical task outcomes',
                             'no root/action reselection or physical benefit claim'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planner-output', required=True)
    parser.add_argument('--output', required=True)
    a = parser.parse_args()
    result = audit(a.planner_output)
    with Path(a.output).open('xb') as stream:
        stream.write(json.dumps(result, sort_keys=True, allow_nan=False).encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result['status'], sha256=sha(a.output),
                         disagreements=sum(r['diagnostic_ranking_disagrees'] for r in result['rows']))))


if __name__ == '__main__':
    main()
