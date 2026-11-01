"""Synthetic saved arithmetic only; does not evaluate a planner."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import audit_moving_compiler_surrogate_MAIN_20261027_v1 as m


def fixture():
    goal = np.zeros(192)
    goal[0] = 1
    task = dict(goal=goal.tolist(), projection=np.eye(192).tolist())
    candidates = []
    for i in range(10):
        end = np.zeros(192)
        end[0] = i/10
        candidates.append(dict(index=i, completed=True, independent_model_replay_exact=True,
            predicted_prefix=np.zeros(192).tolist(), predicted_terminal=end.tolist(),
            nominal_terminal=np.zeros(192).tolist(), jacobian=np.zeros((192, 10)).tolist(),
            suffix_params=np.zeros(10).tolist(), stage_cost=1/192,
            nonlinear_terminal_cost=float(np.mean((end-goal)**2)), objective=2/192,
            suffix_solve=dict(linear_terminal_cost=1/192, ridge_penalty=0.)))
    return dict(candidates=candidates, winner=deepcopy(candidates[0])), task


def test_declared_linear_winner_not_rewritten_by_nonlinear_diagnostic():
    result, task = fixture()
    original = deepcopy(result)
    row = m.bank(result, task)
    assert result == original and row['original_index'] == 0 and row['diagnostic_analog_index'] == 9
    assert row['commands_released'] is False and row['diagnostic_ranking_disagrees'] is True


@pytest.mark.parametrize('fault', ['objective', 'terminal', 'winner', 'count', 'jacobian'])
def test_saved_arithmetic_contradictions_rejected(fault):
    r, t = fixture()
    if fault == 'objective': r['candidates'][0]['objective'] += 1
    if fault == 'terminal': r['candidates'][0]['predicted_terminal'][0] += 1
    if fault == 'winner': r['winner'] = deepcopy(r['candidates'][1])
    if fault == 'count': r['candidates'].pop()
    if fault == 'jacobian': r['candidates'][0]['jacobian'] = [[0.]]
    with pytest.raises((AssertionError, ValueError)):
        m.bank(r, t)
