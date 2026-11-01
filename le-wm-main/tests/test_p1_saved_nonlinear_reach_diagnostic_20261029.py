import json
import numpy as np
import pytest

from experiments.spectral_reachability import p1_saved_nonlinear_reach_diagnostic_20261029 as diagnostic

score_candidate = diagnostic.score_candidate


def candidate():
    j = np.zeros((192, 10))
    j[0, 0] = 2
    x = np.zeros(10)
    x[0] = 0.5
    action = np.tanh(x.astype(np.float32)).reshape(5, 2)
    endpoint = np.zeros(192)
    endpoint[0] = 0.25
    return dict(signal=dict(nominal_context=dict(latent_history=[np.zeros(192).tolist()]), jacobian=j.tolist()),
                end=dict(latent_history=[endpoint.tolist()]), suffix_solve=dict(params=x.tolist()), suffix=action.tolist())


def test_scores_use_actual_action_and_actual_endpoint():
    c = candidate()
    goal = np.zeros(192)
    goal[0] = 1
    result = score_candidate(goal, c)
    penalty = .002*.25/192
    assert result['linear_logits'] == pytest.approx(penalty)
    assert result['linear_sent'] == pytest.approx((1-2*np.tanh(np.float32(.5)))**2/192+penalty)
    assert result['nonlinear_sent'] == pytest.approx(.75**2/192+penalty)
    assert result['terminal_q'] == pytest.approx(.75**2/192)


def test_rejects_nonfinite_endpoint():
    c = candidate()
    c['end']['latent_history'][-1][0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        score_candidate(np.zeros(192), c)


def test_panel_summary_is_json_serializable(monkeypatch):
    monkeypatch.setattr(diagnostic.audit, 'analyze', lambda *args: None)
    goal = np.zeros(192)
    goal[0] = 1
    c = candidate()
    scores = score_candidate(goal, c)
    c.update(start=0, iteration=0, terminal_Q=scores['terminal_q'], own_reach_value=scores['linear_logits'])
    report = {'rows': [dict(slot=i//2, arm=str(i%2), result=dict(candidates=[c]*20)) for i in range(8)]}
    prepared = {'requests': [dict(task=dict(goal=goal.tolist())) for _ in range(4)]}
    result = diagnostic.analyze(report, {}, prepared)
    assert result['summary']['candidates'] == 160
    json.dumps(result, allow_nan=False)
