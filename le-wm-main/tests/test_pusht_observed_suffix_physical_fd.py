import numpy as np
import pytest
from experiments.spectral_reachability import run_pusht_observed_suffix_physical_fd as fd


def test_fixed_probes():
    probes = fd.probes()
    assert len(probes) == 9
    assert probes[0]['params'] == [0., 0.]
    assert {p['epsilon'] for p in probes[1:]} == {.01, .05}
    for p in probes[1:]:
        assert p['params'][p['axis']] == p['sign'] * p['epsilon']


def test_only_original_selected_and_zero():
    row = {'selected_candidate_id': 'prefix_9', 'outcome_best': 'prefix_2'}
    assert fd.prefix_ids(row) == ['prefix_9', 'prefix_0']
    with pytest.raises(ValueError):
        fd.prefix_ids({'selected_candidate_id': 'prefix_0'})


def records():
    jac = np.array([[1., 2.], [3., -1.]])
    return jac, [dict(p, status='executed', vector=(jac @ np.asarray(p['params'], np.float32).astype(float)).tolist()) for p in fd.probes()]


def test_central_fd_matches_linear_fixture_only():
    jac, rows = records()
    result = fd.compare(rows, jac, np.eye(2))
    assert len(result) == 2
    for row in result:
        np.testing.assert_allclose(row['physical_jacobian'], jac)
        assert row['raw_mse'] < 1e-25


def test_failed_probe_not_dropped_or_replaced():
    jac, rows = records()
    rows[1] = dict(rows[1], status='failed', error='fixture failure')
    result = fd.compare(rows, jac, np.eye(2))
    assert result[0]['status'] == 'unavailable'
    assert result[1]['status'] == 'compared'
    assert len(rows) == 9


@pytest.mark.parametrize('jac,q', [([[True, 0]], [[1]]), ([[1, 2]], [[float('nan')]]), ([[1]], [[1]])])
def test_invalid_saved_arrays(jac, q):
    with pytest.raises(ValueError):
        fd.compare([], jac, q)


def test_missing_duplicate_and_nonfinite_probe_preserve_two_epsilon_rows():
    jac, rows = records()
    for changed in (rows[1:], rows + [rows[1]], [dict(r, vector=[float('nan'), 0]) for r in rows]):
        result = fd.compare(changed, jac, np.eye(2))
        assert len(result) == 2
        # Removing just zero does not remove any central difference pair.
        assert result[0]['status'] == ('compared' if changed == rows[1:] else 'unavailable')


def test_no_outcome_dependent_selection_and_bounds():
    assert len(fd.probes()) * 4 * len(fd.prefix_ids({'selected_candidate_id': 'prefix_3'})) == 72
    assert all(len(p['params']) == 2 and max(abs(x) for x in p['params']) <= .05 for p in fd.probes())


def test_runner_seals_before_first_cuda_and_never_evaluates_saved_J():
    import inspect
    source = inspect.getsource(fd.run)
    assert source.index('seal = create_seal') < source.index("host = base.require_cuda_allocation")
    assert '_suffix_terminal(' not in source and '_encode_context(' not in source
    assert "model.predict = encode, predict" in source
