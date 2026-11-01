import numpy as np
import pytest
from experiments.spectral_reachability import main_nonlinear_suffix_history_control_20261028_v1 as m


def context():
    root = np.stack([np.full(192, x, np.float32) for x in (0, 1, 2)])
    mid = np.concatenate((root[1:], np.full((1, 192), 3, np.float32)))
    return root, mid


def test_exact_old2_changed_last_not_claimed_last_only():
    root, mid = context()
    obs = mid.copy(); obs[-1] += 1
    r = m.history_diagnostics(root, mid, obs)
    assert r['matched_old2_exact'] and r['rows'][2]['MSE'] == 1
    assert not r['last_only_intervention'] and not r['drift_cause_attributed']


def test_old2_reencoding_drift_retained_without_tolerance():
    root, mid = context()
    obs = mid.copy(); obs[0, 0] = np.nextafter(obs[0, 0], np.float32(2))
    r = m.history_diagnostics(root, mid, obs)
    assert not r['matched_old2_exact']
    assert r['rows'][0]['MSE'] > 0


def test_wrong_imagined_temporal_shift_refused():
    root, mid = context()
    with pytest.raises(ValueError):
        m.history_diagnostics(root, np.roll(mid, 1, axis=0), mid)


def test_missing_history_refused():
    root, mid = context()
    with pytest.raises(ValueError):
        m.history_diagnostics(root, mid[:2], mid)
