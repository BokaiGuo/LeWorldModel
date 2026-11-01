"""Scalar audit tests, no neural callbacks."""
from copy import deepcopy
import pytest
from experiments.spectral_reachability import main_p3_secant_gate_scalar_20261029_v1 as m


def sample():
    goal = [0.]*192
    item = dict(status='complete', immediate=1., reach=.25, index=0, radius=.1,
        postcontext=dict(latent_history=[[1.]*192]), signal=dict(nominal=[.5]*192, jacobian=[[0.]*10 for _ in goal]),
        solve=dict(params=[0.]*10, linear_terminal_cost=.25, ridge_penalty=0., own_reach_value=.25),
        nongreedy=True, better_reach=True, eligible=True)
    p = dict(tasks=[dict(goal=goal, projection=[[float(i == j) for j in range(192)] for i in range(192)])]*2,
             source_groups=[])
    r = dict(schema=m.PRODUCER+'.report', status='four_groups_settled_NOT_delivery', errors=[], delivery=None, rows=[])
    c = dict(schema=m.PRODUCER+'.consume', groups=4, candidate_denominator=16,
             status='all4_actual_cp5_secant_groups_verified_NOT_delivery', arms=[])
    for slot, (root, h) in enumerate(((0, 1), (0, 3), (1, 1), (1, 3))):
        identity = dict(slot=slot, root=root, horizon=h)
        p['source_groups'].append(dict(**identity, zero=dict(immediate=.5, reach=.5)))
        rows = [dict(deepcopy(item), index=i) for i in range(4)]
        result = dict(status='all_four_scored', errors=[], delivery=None, rows=rows, eligible_indices=list(range(4)))
        r['rows'].append(dict(**identity, status='all_four_scored', result=result))
        c['arms'].append(dict(**identity, status='all_four_scored', complete=True, eligible_indices=list(range(4))))
    return p, r, c


def test_all_sixteen_and_original_gate():
    out = m.explain(*sample())
    assert out['denominator'] == 16 and all(g['eligible_indices'] == list(range(4)) for g in out['groups'])
    assert out['new_neural_calls'] == out['new_physics_calls'] == 0 and out['delivery'] is None


@pytest.mark.parametrize('fault', ['root', 'index', 'score', 'gate', 'receipt', 'Q', 'missing'])
def test_reject_source_score_and_gate_changes(fault):
    p, r, c = sample()
    if fault == 'root': p['source_groups'][0]['root'] = 1
    elif fault == 'index': r['rows'][0]['result']['rows'][0]['index'] = 1
    elif fault == 'score': r['rows'][0]['result']['rows'][0]['reach'] += .1
    elif fault == 'gate': r['rows'][0]['result']['rows'][0]['eligible'] = False
    elif fault == 'receipt': c['arms'][0]['eligible_indices'] = []
    elif fault == 'Q': p['tasks'][0]['projection'][0][0] = 2.
    else: r['rows'].pop()
    with pytest.raises(ValueError): m.explain(p, r, c)


def test_exact_tie_rejected_even_when_scalar_reconstruction_is_close():
    p, r, c = sample()
    p['source_groups'][0]['zero']['immediate'] = 1.
    for x in r['rows'][0]['result']['rows']: x.update(nongreedy=False, eligible=False)
    r['rows'][0]['result']['eligible_indices'] = []; c['arms'][0]['eligible_indices'] = []
    assert m.explain(p, r, c)['groups'][0]['eligible_indices'] == []
