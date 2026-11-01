import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_joint_vreach_saved_diff_20261029_v1 as m


def test_independent_ridge_active_and_inactive():
    j = np.eye(2); r = np.array([2., 0.])
    x, mu = m.independent_solve(j, r, .002, 1.)
    assert np.linalg.norm(x) == pytest.approx(1.) and mu > 0
    assert np.linalg.norm((j.T@j+(.002+mu)*np.eye(2))@x-j.T@r) < 1e-10
    y, mu = m.independent_solve(j, r, .002, 5.)
    assert mu == 0 and y[0] == pytest.approx(2/1.002)


def test_raw_vs_tanh_counterfactual_and_sign():
    j = np.eye(2); target = np.array([2., 0.])
    x, _ = m.independent_solve(j, target, .002, 1.)
    d = m.compare(j, target, x, .002)
    assert d['score_raw'] < d['score_tanh_raw']
    assert d['raw_vs_tanh_l2'] > 0
    assert d['negative_sign_score'] > d['score_raw']


def test_crossed_context_decomposes_jacobian_and_residual():
    baseline_j = np.eye(2); candidate_j = 2*np.eye(2)
    base_r = np.ones(2); candidate_r = np.ones(2)*2
    d = m.crossed_scores(baseline_j, base_r, candidate_j, candidate_r, .002, 1.)
    assert d['candidate_j_base_residual'] < d['baseline_j_base_residual']
    assert d['baseline_j_candidate_residual'] > d['baseline_j_base_residual']


def test_action_normalization_and_context_binding():
    task = {'action_mean': [0., 0.], 'action_std': [2., 4.]}
    old = {'action_history': [[1.]*10, [2.]*10]}
    raw = [[0.5, -0.5]]*5
    next_context = {'action_history': [[2.]*10, [0.25, -0.125]*5]}
    m.check_transition(task, old, raw, next_context)
    next_context['action_history'][1][0] += 0.1
    with pytest.raises(ValueError): m.check_transition(task, old, raw, next_context)


def test_exclusive_write_and_external_hash(tmp_path):
    out = tmp_path/'diagnostic.json'
    receipt = m.write(out, {'a': 1})
    assert m.read_verified(out, receipt) == {'a': 1}
    with pytest.raises(FileExistsError): m.write(out, {'a': 2})
    with pytest.raises(ValueError): m.read_verified(out, '0'*64)
