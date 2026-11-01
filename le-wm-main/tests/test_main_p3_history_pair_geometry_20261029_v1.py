from copy import deepcopy
import pytest
from experiments.spectral_reachability import main_p3_history_pair_geometry_20261029_v1 as m


def point(value, label=0., goal=1.):
    p, a, g = [[x]*192 for x in (value, label, goal)]
    return dict(predicted192=p, actual192=a, goal192=g, **m.geometry(p, a, g))


def fixture():
    result = []
    for i in range(8):
        branch, time = divmod(i, 2)
        result.append(dict(index=i, root_slot=branch//2, bank=('endpoints', 'midpoints')[branch % 2],
                           start=(10, 15)[time], end=(15, 20)[time], A=point(.1), R=point(.2),
                           same_physical_label=True, physical_benefit_claim=False))
    return result


def test_geometry_cancellation_and_zero():
    g = m.geometry([.5]*192, [0.]*192, [1.]*192)
    assert g['error_mse'] == .25 and g['actual_residual_cross_term'] == -1.
    assert g['model_minus_actual_Q'] == -.75 and g['squared_error_cross_cancel']
    assert m.geometry([0.]*192, [0.]*192, [0.]*192)['identity_residual'] == 0


def test_all_slots_pairs_and_no_epsilon_ties():
    r = m.describe(fixture())
    assert r['counts'] == dict(endpoints=8, paired_comparisons=4, A_sign_matches=4,
                              R_sign_matches=4, A_cross_cancel=8, R_cross_cancel=8)
    assert all(p['actual_sign'] == 0 for p in r['pairs'])


@pytest.mark.parametrize('fault', ('missing', 'order', 'time', 'label', 'goal', 'prediction', 'nonfinite', 'score', 'claim'))
def test_fail_closed(fault):
    r = deepcopy(fixture())
    if fault == 'missing': r.pop()
    elif fault == 'order': r.reverse()
    elif fault == 'time': r[0]['end'] = 20
    elif fault == 'label': r[0]['R']['actual192'][0] = 1.
    elif fault == 'goal': r[0]['R']['goal192'][0] = 2.
    elif fault == 'prediction': r[0]['A']['predicted192'].pop()
    elif fault == 'nonfinite': r[0]['A']['predicted192'][0] = float('nan')
    elif fault == 'score': r[0]['A']['error_mse'] = 42.
    else: r[0]['physical_benefit_claim'] = True
    with pytest.raises(ValueError): m.describe(r)


def test_pair_flip_is_preserved_not_dropped():
    r = fixture()
    for i in (0, 1, 4, 5):
        r[i]['A'] = point(.3, label=.2)
        r[i]['R'] = point(-.2, label=.2)
    out = m.describe(r)
    assert out['counts']['A_sign_matches'] == 4
    assert out['counts']['R_sign_matches'] == 0
