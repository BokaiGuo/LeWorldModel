"""Independent saved-metric regression, no neural imports or evidence."""
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import consume_capability_direction_MAIN_20261027_v1 as m


def test_zero_negative_error_and_cross_term_preserved():
    p, a, g = np.ones(192), np.zeros(192), np.zeros(192)
    r = m.aligned(p,a,g)
    assert r == dict(predicted_Q=1., actual_Q=0., prediction_MSE=1., cross_term=-2.)
    assert m.aligned(a,a,g)['prediction_MSE'] == 0


@pytest.mark.parametrize('bad', [np.zeros(191), np.full(192,np.nan), np.full(192,np.inf)])
def test_bad_latents_refused(bad):
    with pytest.raises(ValueError): m.aligned(bad,np.zeros(192),np.zeros(192))


def test_all_ten_pairs_keep_actual_ties_and_prediction_reversals():
    rows = [dict(probe=p, actual_Q=a, teacher=dict(predicted_Q=b)) for p,a,b in
            zip(('zero','+x','-x','+y','-y'),(0.,0.,1.,-1.,2.),(0.,1.,-1.,1.,2.))]
    r = m.rank_pairs(rows,'teacher')
    assert len(r) == 10
    assert r[0]['actual_tie'] is True and r[0]['exact_sign_agreement'] is False
    assert r[1]['exact_sign_agreement'] is False
    assert r[-1]['left'] == '+y' and r[-1]['right'] == '-y'


def test_json_duplicate_nan_rejected(tmp_path):
    f = tmp_path/'bad.json'
    for payload in ('{"a":1,"a":2}', '{"a":NaN}'):
        f.write_text(payload)
        with pytest.raises(ValueError): m.read(f)


def test_consumer_does_not_import_neural_or_physics_runtime():
    assert not any(name in sys.modules for name in ('torch','gymnasium','pymunk','stable_worldmodel'))
