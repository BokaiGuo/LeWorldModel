import json
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability.run_pusht_program_search_cuda import metric_for
from experiments.spectral_reachability.pusht_d2p_nuisance_program_scoring import score_terminal_predictions


def test_saved_metric_matches_compiler_all_roots():
    source = Path('results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json')
    if not source.exists():
        pytest.skip('local sealed calibration artifact absent')
    for row in json.loads(source.read_text())['rows']:
        c = row['compiled']
        vectors = [r['predicted_terminal_vector'] for r in c['receipts']]
        expected = score_terminal_predictions(goal_vectors=c['goal_vectors'], terminal_vectors=vectors,
            projection=c['scoring']['metric']['projection'])['predicted_terminal_costs']
        actual = [metric_for(c)({'latent_history': [v]}) for v in vectors]
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_goal_variants_not_averaged_as_tasks():
    c = {'goal_vectors': [np.zeros(192).tolist(), np.ones(192).tolist()],
         'scoring': {'metric': {'projection': np.eye(192).tolist()}}}
    assert metric_for(c)({'latent_history': [np.zeros(192)]}) == 0
    assert metric_for(c)({'latent_history': [np.ones(192)]}) == 1


def test_nonfinite_metric_rejected():
    c = {'goal_vectors': [[float('nan')]*192],
         'scoring': {'metric': {'projection': np.eye(192).tolist()}}}
    with pytest.raises(ValueError):
        metric_for(c)
