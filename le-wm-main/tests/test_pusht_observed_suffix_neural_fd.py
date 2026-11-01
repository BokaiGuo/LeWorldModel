"""No neural CPU execution: array arithmetic/call-loop and real report contracts."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import run_pusht_observed_suffix_neural_fd as runner


def samples():
    j = np.array([[1., 2.], [-3., 4.], [5., -.5]])
    origin = np.array([.4, .5, .6])
    values = []
    for probe in runner.probes():
        params = np.asarray(probe['params'], np.float32).astype(float)
        values.append(dict(probe, executed_params=params.tolist(), status='executed', vector=(origin+j@params).tolist()))
    return values, dict(jacobian=j.tolist(), projection=np.eye(3).tolist(), nominal=origin.tolist())


def test_real_call_loop_uses_every_saved_parameter_once_and_saves_all_outputs():
    physical, plan = samples()
    calls, saved = [], []
    def terminal(params):
        # Array-only callable, not a neural model or CPU fallback.
        calls.append(params.copy())
        return np.asarray(plan['nominal'])+np.asarray(plan['jacobian'])@params
    outputs = runner.evaluate_probes(terminal, np.asarray, np.asarray, physical, saved.append)
    assert len(calls) == len(outputs) == len(saved) == 9
    assert [p.tolist() for p in calls] == [p['executed_params'] for p in physical]
    for comparison in runner.comparisons(outputs, physical, plan):
        assert comparison['neural_vs_saved_ad']['mse'] < 1e-25
        assert comparison['neural_vs_physical']['mse'] == 0


def test_parameter_denominator_includes_float32_rounding():
    physical, plan = samples()
    for epsilon in (.01, .05):
        np.testing.assert_allclose(runner.central_jacobian(physical, epsilon), plan['jacobian'], atol=1e-13)


def test_zero_physical_direction_keeps_cosine_undefined():
    result = runner.matrix_comparison(np.ones((3, 2)), np.zeros((3, 2)))
    assert result['flattened_cosine'] is None and result['relative_error_to_right'] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('damage', ['duplicate', 'params', 'status', 'vector', 'count', 'executed'])
def test_probe_corruption_fails_closed(damage):
    values, _ = samples()
    if damage == 'duplicate':
        values[1] = copy.deepcopy(values[2])
    elif damage == 'params':
        values[1]['params'] = [.3, 0.]
    elif damage == 'status':
        values[1]['status'] = 'failed'
    elif damage == 'vector':
        values[1]['vector'][0] = float('nan')
    elif damage == 'count':
        values.pop()
    else:
        values[1]['executed_params'][0] = -.011
    with pytest.raises(ValueError):
        runner.validate_probes(values, 3)


def test_duplicate_identity_rejected():
    with pytest.raises(ValueError):
        runner.keyed([{'id': 'a'}, {'id': 'a'}], 'id')


def test_call_loop_does_not_hide_prediction_failure():
    values, _ = samples()
    def fail(params):
        raise RuntimeError('inference failed')
    with pytest.raises(RuntimeError, match='inference failed'):
        runner.evaluate_probes(fail, np.asarray, np.asarray, values, lambda value: None)


def test_import_and_help_do_not_load_torch_or_allocate():
    code = ('import sys; from experiments.spectral_reachability import run_pusht_observed_suffix_neural_fd; '
            'assert "torch" not in sys.modules')
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True)
    result = subprocess.run([sys.executable, '-m', runner.__name__, '--help'], check=True, capture_output=True, text=True)
    assert '--physical-fd' in result.stdout and '--policy' in result.stdout


def test_actual_saved_physical_report_probe_and_plan_contracts():
    path = Path(__file__).resolve().parents[1]/'results/pusht_observed_suffix_physical_fd4_v1_20261026_cuda/report.json'
    if not path.is_file():
        pytest.skip('local calibration artifacts unavailable')
    report = json.loads(path.read_text())
    runner._completed(report)
    assert report['schema'] == 'pusht_observed_suffix_physical_fd.v1'
    assert report['protocol']['probes'] == runner.probes()
    observed_paths = [Path(p) for p in report['protocol']['bindings'] if 'observed_oneshot_suffix_control' in p and Path(p).name == 'report.json']
    assert len(observed_paths) == 1
    observed = json.loads(observed_paths[0].read_text())
    observed_rows = runner.keyed(observed['rows'], 'root_sha256')
    count = 0
    for row in report['rows']:
        observed_row = observed_rows[row['root_sha256']]
        assert set(runner.keyed(row['branches'], 'candidate_id')) == set(runner.prefix_ids(observed_row))
        for branch in row['branches']:
            source = runner.keyed(observed_row['branches'], 'candidate_id')[branch['candidate_id']]
            plan = json.loads(Path(source['plan_path']).read_text())
            runner.validate_probes(branch['probes'], len(plan['nominal']))
            assert np.array_equal(plan['jacobian'], branch['saved_model_jacobian'])
            assert np.array_equal(plan['nominal'], branch['saved_model_zero_nominal'])
            # Exercise actual production comparison on saved vectors; no inference.
            result = runner.comparisons(branch['probes'], branch['probes'], plan)
            assert all(v['neural_vs_physical']['mse'] == 0 for v in result)
            count += 1
    assert count == 8
